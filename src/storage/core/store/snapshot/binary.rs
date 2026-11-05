//! Versioned binary snapshot envelope. Column payloads are ALREADY archived;
//! move them into an rkyv envelope instead of JSON-expanding every byte.
//! This is a codec primitive, not a journal, disk policy or recovery driver.
use super::*;
use crate::storage::column::ColumnCodecId;

const MAGIC: &[u8; 8] = b"ENSSNP01";
const PREFIX: usize = 16; // magic + exact archive byte length; root stays aligned.

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Envelope {
    version: u32,
    slots: Vec<(u32, Option<u128>, Option<(u32, u32)>)>,
    free: Vec<u32>,
    archetypes: Vec<ArchivedTable>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchivedTable {
    id: u32,
    types: Vec<u64>,
    entities: Vec<u128>,
    columns: Vec<ArchivedPayload>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchivedPayload {
    id: u64,
    component: u64,
    schema: u32,
    rows: u32,
    codec: u8,
    bytes: Vec<u8>,
}

fn codec_error(message: impl Into<String>) -> StateError {
    StateError::CodecError(message.into())
}

impl ColumnarStoreSnapshot {
    /// Consume the existing archive payloads; never encode native T again or
    /// convert payload bytes into text. Ordinary serde snapshots remain useful
    /// for inspection; the reliable participant uses this versioned envelope.
    pub fn into_binary(self) -> Result<Vec<u8>, StateError> {
        if self.format_version != CURRENT_SNAPSHOT_VERSION {
            return Err(codec_error("Unsupported binary snapshot version"));
        }
        let envelope = Envelope {
            version: self.format_version,
            slots: self
                .allocator
                .snapshot_slots()
                .map(|(g, id, loc)| (g, id.map(|id| id.to_u128()), loc))
                .collect(),
            free: self.allocator.snapshot_free_slots().to_vec(),
            archetypes: self
                .archetypes_data
                .into_iter()
                .map(|table| ArchivedTable {
                    id: table.archetype_id,
                    types: table
                        .component_types
                        .into_iter()
                        .map(|id| id.as_u64())
                        .collect(),
                    entities: table.entities.into_iter().map(|id| id.to_u128()).collect(),
                    columns: table
                        .columns
                        .into_iter()
                        .map(|(id, column)| ArchivedPayload {
                            // Validate the directory key separately; don't silently
                            // canonicalize a forged component ID into a valid header.
                            id: id.as_u64(),
                            component: column.component_type_id.as_u64(),
                            schema: column.schema_version,
                            rows: column.row_count,
                            codec: column.codec as u8,
                            bytes: column.payload,
                        })
                        .collect(),
                })
                .collect(),
        };
        let archived = rkyv::to_bytes::<rkyv::rancor::BoxedError>(&envelope)
            .map_err(|error| codec_error(format!("Snapshot envelope encoding: {error}")))?;
        let len = u64::try_from(archived.len())
            .map_err(|_| codec_error("Snapshot length exceeds u64"))?;
        let capacity = PREFIX
            .checked_add(archived.len())
            .ok_or_else(|| codec_error("Snapshot length overflow"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| StateError::CapacityExhausted("Snapshot envelope buffer".into()))?;
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&archived);
        Ok(bytes)
    }

    /// Bytecheck validates the envelope BEFORE reconstructing metadata. A
    /// caller's Vec/u8 slice may be unaligned: make an explicitly aligned copy,
    /// never unchecked-cast it. Native column/schema/layout validation remains
    /// in restore_snapshot, before any Current installation.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, StateError> {
        if bytes.len() < PREFIX || &bytes[..8] != MAGIC {
            return Err(codec_error("Invalid binary snapshot envelope magic/header"));
        }
        let len = u64::from_le_bytes(bytes[8..PREFIX].try_into().unwrap());
        let len = usize::try_from(len)
            .map_err(|_| codec_error("Snapshot length exceeds address space"))?;
        if len != bytes.len() - PREFIX {
            return Err(codec_error("Snapshot envelope length mismatch"));
        }
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(&bytes[PREFIX..]);
        let envelope = rkyv::from_bytes::<Envelope, rkyv::rancor::BoxedError>(&aligned)
            .map_err(|error| codec_error(format!("Snapshot envelope validation: {error}")))?;
        if envelope.version != CURRENT_SNAPSHOT_VERSION {
            return Err(codec_error("Unsupported binary snapshot version"));
        }
        let allocator = EntityAllocator::from_snapshot_parts(
            envelope
                .slots
                .into_iter()
                .map(|(g, id, loc)| (g, id.map(EntityId::from_u128), loc))
                .collect(),
            envelope.free,
        )?;
        let mut archetypes_data = Vec::with_capacity(envelope.archetypes.len());
        for table in envelope.archetypes {
            let mut columns = BTreeMap::new();
            for column in table.columns {
                let codec = match column.codec {
                    0 => ColumnCodecId::None,
                    1 => ColumnCodecId::Rkyv,
                    2 => ColumnCodecId::RawBytes,
                    3 => ColumnCodecId::ArrowColumnar,
                    _ => return Err(codec_error("Unknown binary column codec")),
                };
                let id = ComponentTypeId::new(column.id);
                let value = ArchivedColumnData::new(
                    ComponentTypeId::new(column.component),
                    column.schema,
                    column.rows,
                    codec,
                    column.bytes,
                );
                if columns.insert(id, value).is_some() {
                    return Err(codec_error("Duplicate binary snapshot column"));
                }
            }
            archetypes_data.push(ArchetypeSnapshotData {
                archetype_id: table.id,
                component_types: table.types.into_iter().map(ComponentTypeId::new).collect(),
                entities: table
                    .entities
                    .into_iter()
                    .map(EntityId::from_u128)
                    .collect(),
                columns,
            });
        }
        Ok(Self {
            format_version: envelope.version,
            allocator,
            archetypes_data,
        })
    }
}


#[cfg(test)]
#[path = "tests.rs"]
mod tests;
