//! Columnar transport boundary. No disk I/O, simulation clock or timeline policy.
use super::{BatchOp, EntityBundle, EntityPatch, PatchOp, StoreBatch};
use crate::storage::core::registry::ComponentRegistry;
use crate::storage::{ArchivedColumnData, ColumnCodecId, ErasedComponentValue};
use crate::{ComponentTypeId, EntityId, StateError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ENCODED_BATCH_FORMAT_V2: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EncodedColumnOpKind {
    Insert,
    Replace,
    Upsert,
}

#[derive(Debug, Clone)]
pub struct EncodedColumnRun {
    /// Payload row i always belongs to entities[i], independent of Current rows.
    pub entities: Vec<EntityId>,
    /// Optional zero-copy shared entity slice used by background archive encoding.
    pub shared_entities: Option<std::sync::Arc<[EntityId]>>,
    pub op_kind: EncodedColumnOpKind,
    pub archived: ArchivedColumnData,
}

impl PartialEq for EncodedColumnRun {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.op_kind == other.op_kind
            && self.archived == other.archived
            && self.same_entities(other)
    }
}

impl Eq for EncodedColumnRun {}

use crate::storage::core::entity_index::entity_slice_serde;

impl Serialize for EncodedColumnRun {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("EncodedColumnRun", 3)?;
        state.serialize_field(
            "entities",
            &entity_slice_serde::EntitySliceRef(self.entities_slice()),
        )?;
        state.serialize_field("op_kind", &self.op_kind)?;
        state.serialize_field("archived", &self.archived)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for EncodedColumnRun {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct WireEncodedColumnRun {
            #[serde(deserialize_with = "entity_slice_serde::deserialize")]
            entities: Vec<EntityId>,
            op_kind: EncodedColumnOpKind,
            archived: ArchivedColumnData,
        }
        let wire = WireEncodedColumnRun::deserialize(deserializer)?;
        Ok(Self {
            entities: wire.entities,
            shared_entities: None,
            op_kind: wire.op_kind,
            archived: wire.archived,
        })
    }
}

impl EncodedColumnRun {
    #[inline]
    pub fn entities_slice(&self) -> &[EntityId] {
        if !self.entities.is_empty() {
            &self.entities
        } else if let Some(shared) = &self.shared_entities {
            shared
        } else {
            &self.entities
        }
    }

    #[inline]
    pub fn estimated_entities_wire_bytes(&self) -> usize {
        let slice = self.entities_slice();
        if slice.len() >= 4 {
            let first = slice[0].to_u128();
            let second = slice[1].to_u128();
            let third = slice[2].to_u128();
            let last = slice[slice.len() - 1].to_u128();
            let stride = second.wrapping_sub(first);
            if third == second.wrapping_add(stride)
                && last == first.wrapping_add(stride.wrapping_mul((slice.len() - 1) as u128))
            {
                return 64;
            }
            if Self::has_disjoint_positive_stride_runs(slice) {
                return 128;
            }
            if (first >> 64) == (last >> 64) {
                return slice.len().saturating_mul(8).saturating_add(64);
            }
        }
        slice.len().saturating_mul(16).saturating_add(16)
    }

    #[inline]
    pub(crate) fn entity_len(&self) -> usize {
        self.entities_slice().len()
    }

    #[inline]
    pub(crate) fn entity_at(&self, index: usize) -> Option<EntityId> {
        self.entities_slice().get(index).copied()
    }

    #[inline]
    pub(crate) fn is_strictly_increasing_non_nil(&self) -> bool {
        let slice = self.entities_slice();
        !slice.iter().any(|id| id.is_nil()) && slice.windows(2).all(|w| w[0] < w[1])
    }

    #[inline]
    pub(crate) fn has_distinct_non_nil_entities(&self) -> bool {
        let slice = self.entities_slice();
        if !slice.iter().any(|id| id.is_nil()) && slice.windows(2).all(|w| w[0] < w[1]) {
            return true;
        }
        Self::has_disjoint_positive_stride_runs(slice)
    }

    #[inline]
    fn has_disjoint_positive_stride_runs(slice: &[EntityId]) -> bool {
        if slice.len() < 4 {
            return false;
        }
        let mut runs: [(u128, u128, u128); 4] = [(0, 0, 0); 4];
        let mut run_count = 0usize;
        let mut idx = 0usize;
        while idx < slice.len() {
            if run_count >= 4 {
                return false;
            }
            let first = slice[idx];
            if first.is_nil() {
                return false;
            }
            let first_u128 = first.to_u128();
            if idx + 1 >= slice.len() {
                runs[run_count] = (first_u128, first_u128, 1);
                run_count += 1;
                break;
            }
            let second = slice[idx + 1];
            if second.is_nil() || second <= first {
                return false;
            }
            let stride = second.to_u128() - first_u128;
            let mut prev_u128 = second.to_u128();
            let mut cur = idx + 2;
            while cur < slice.len() {
                let next = slice[cur];
                if next.is_nil() {
                    return false;
                }
                let next_u128 = next.to_u128();
                if next_u128 != prev_u128.wrapping_add(stride) || next_u128 <= prev_u128 {
                    break;
                }
                prev_u128 = next_u128;
                cur += 1;
            }
            runs[run_count] = (first_u128, prev_u128, stride);
            run_count += 1;
            idx = cur;
        }
        for i in 0..run_count {
            let (first_a, last_a, stride_a) = runs[i];
            for &(first_b, last_b, stride_b) in &runs[(i + 1)..run_count] {
                if last_a < first_b || last_b < first_a {
                    continue;
                }
                if stride_a > 0 && stride_a == stride_b && first_a.abs_diff(first_b) % stride_a != 0
                {
                    continue;
                }
                return false;
            }
        }
        run_count > 0
    }

    #[inline]
    pub(crate) fn same_entities(&self, other: &Self) -> bool {
        if let (Some(a), Some(b)) = (&self.shared_entities, &other.shared_entities)
            && self.entities.is_empty()
            && other.entities.is_empty()
            && std::sync::Arc::ptr_eq(a, b)
        {
            return true;
        }
        self.entities_slice() == other.entities_slice()
    }
}

/// A logical batch, not a receipt or timeline edge by itself. Every field is
/// untrusted at decode. Older per-cell wire formats are not guessed or accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedStoreBatch {
    pub format_version: u32,
    pub created_entities: Vec<EntityId>,
    pub deleted_entities: Vec<EntityId>,
    pub removed_components: Vec<(EntityId, ComponentTypeId)>,
    pub column_runs: Vec<EncodedColumnRun>,
}

impl Default for EncodedStoreBatch {
    fn default() -> Self {
        Self {
            format_version: ENCODED_BATCH_FORMAT_V2,
            created_entities: Vec::new(),
            deleted_entities: Vec::new(),
            removed_components: Vec::new(),
            column_runs: Vec::new(),
        }
    }
}

impl EncodedStoreBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.created_entities.is_empty()
            && self.deleted_entities.is_empty()
            && self.removed_components.is_empty()
            && self.column_runs.is_empty()
    }

    /// 检查当前编码批次是否对目标组件类型 `cid` 产生修改（含列运行覆盖/新增、组件移除或实体删除）。
    ///
    /// 用于 L2 历史链按单组件重放时的表级短路：若返回 `false`，无需解码该批次中的任何列数据。
    #[inline]
    pub fn affects_component(&self, cid: ComponentTypeId) -> bool {
        !self.deleted_entities.is_empty()
            || self.removed_components.iter().any(|&(_, c)| c == cid)
            || self
                .column_runs
                .iter()
                .any(|run| run.archived.component_type_id == cid)
    }

    /// Number of distinct logical entity targets, not column runs or cells.
    pub fn len(&self) -> usize {
        self.created_entities
            .iter()
            .chain(&self.deleted_entities)
            .copied()
            .chain(self.removed_components.iter().map(|&(id, _)| id))
            .chain(
                self.column_runs
                    .iter()
                    .flat_map(|run| (0..run.entity_len()).filter_map(move |i| run.entity_at(i))),
            )
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Complete metadata pass before invoking ANY component deserializer.
    pub(crate) fn validate_metadata(&self, registry: &ComponentRegistry) -> Result<(), StateError> {
        if self.format_version != ENCODED_BATCH_FORMAT_V2 {
            return Err(StateError::CodecError(format!(
                "Unsupported encoded batch version {} (expected {ENCODED_BATCH_FORMAT_V2})",
                self.format_version
            )));
        }
        let conflict = |reason: &str| StateError::BatchConflict(reason.into());
        if self.created_entities.is_empty()
            && self.deleted_entities.is_empty()
            && self.removed_components.is_empty()
        {
            let mut seen_cids = smallvec::SmallVec::<[ComponentTypeId; 8]>::new();
            let mut distinct_cids = true;
            for run in &self.column_runs {
                let archived = &run.archived;
                if archived.codec != ColumnCodecId::Rkyv {
                    return Err(StateError::CodecError(
                        "Encoded batch column must use Rkyv".into(),
                    ));
                }
                if run.entity_len() != archived.row_count as usize {
                    return Err(StateError::CodecError(
                        "Encoded batch column row count mismatch".into(),
                    ));
                }
                let cid = archived.component_type_id;
                let def = registry
                    .get(&cid)
                    .ok_or_else(|| StateError::TypeNotRegistered(format!("{cid:?}")))?;
                if !def.has_archive_capability() || def.codec() != ColumnCodecId::Rkyv {
                    return Err(StateError::AccessDenied(format!(
                        "Component {cid} has no registered archive capability"
                    )));
                }
                if def.schema_version() != archived.schema_version {
                    return Err(StateError::CodecError(format!(
                        "Schema version mismatch for component {cid}"
                    )));
                }
                if seen_cids.contains(&cid) {
                    distinct_cids = false;
                } else {
                    seen_cids.push(cid);
                }
            }
            if distinct_cids
                && self
                    .column_runs
                    .iter()
                    .all(EncodedColumnRun::has_distinct_non_nil_entities)
            {
                return Ok(());
            }
        }
        let mut created = BTreeSet::new();
        let mut deleted = BTreeSet::new();
        for &id in &self.created_entities {
            if id.is_nil() || !created.insert(id) {
                return Err(conflict("Invalid/duplicate created entity"));
            }
        }
        for &id in &self.deleted_entities {
            if id.is_nil() || !deleted.insert(id) || created.contains(&id) {
                return Err(conflict(
                    "Invalid/duplicate or contradictory deleted entity",
                ));
            }
        }
        let mut targets = BTreeSet::new();
        for &(id, cid) in &self.removed_components {
            if id.is_nil()
                || created.contains(&id)
                || deleted.contains(&id)
                || !targets.insert((id, cid))
            {
                return Err(conflict("Duplicate/contradictory removed component"));
            }
        }
        for run in &self.column_runs {
            let archived = &run.archived;
            if archived.codec != ColumnCodecId::Rkyv {
                return Err(StateError::CodecError(
                    "Encoded batch column must use Rkyv".into(),
                ));
            }
            if run.entity_len() != archived.row_count as usize {
                return Err(StateError::CodecError(
                    "Encoded batch column row count mismatch".into(),
                ));
            }
            let cid = archived.component_type_id;
            let def = registry
                .get(&cid)
                .ok_or_else(|| StateError::TypeNotRegistered(format!("{cid:?}")))?;
            if !def.has_archive_capability() || def.codec() != ColumnCodecId::Rkyv {
                return Err(StateError::AccessDenied(format!(
                    "Component {cid} has no registered archive capability"
                )));
            }
            if def.schema_version() != archived.schema_version {
                return Err(StateError::CodecError(format!(
                    "Schema version mismatch for component {cid}"
                )));
            }
            for i in 0..run.entity_len() {
                let id = run.entity_at(i).unwrap();
                if id.is_nil()
                    || !targets.insert((id, cid))
                    || deleted.contains(&id)
                    || (created.contains(&id) && run.op_kind != EncodedColumnOpKind::Insert)
                {
                    return Err(conflict("Duplicate/contradictory encoded column target"));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn decode_with_registry(
        &self,
        registry: &ComponentRegistry,
    ) -> Result<StoreBatch, StateError> {
        self.validate_metadata(registry)?;
        // All columns decode privately before assembling any state mutation.
        let mut columns = Vec::with_capacity(self.column_runs.len());
        for run in &self.column_runs {
            let def = registry
                .get(&run.archived.component_type_id)
                .expect("metadata validated");
            let mut column = def.create_empty_column();
            column.restore_archived_column(&run.archived)?;
            if column.len() != run.entity_len() {
                return Err(StateError::CodecError(
                    "Decoded column row count mismatch".into(),
                ));
            }
            columns.push(column);
        }
        if self.created_entities.is_empty()
            && self.deleted_entities.is_empty()
            && self.removed_components.is_empty()
            && self.column_runs.len() == 1
        {
            let run = &self.column_runs[0];
            if run.is_strictly_increasing_non_nil() {
                let mut column = columns.pop().unwrap();
                let n = run.entity_len();
                let mut entries = Vec::with_capacity(n);
                for i in (0..n).rev() {
                    let id = run.entity_at(i).unwrap();
                    let value = column.swap_remove_erased(column.len() - 1);
                    let mut patch = EntityPatch::new();
                    match run.op_kind {
                        EncodedColumnOpKind::Insert => {
                            patch.insert_erased(value)?;
                        }
                        EncodedColumnOpKind::Replace => {
                            patch.replace_erased(value)?;
                        }
                        EncodedColumnOpKind::Upsert => {
                            patch.upsert_erased(value)?;
                        }
                    }
                    entries.push((id, BatchOp::PatchEntity { id, patch }));
                }
                entries.reverse();
                let mut batch = StoreBatch::new();
                batch.ops = entries.into_iter().collect();
                return Ok(batch);
            }
        }
        let mut bundles = self
            .created_entities
            .iter()
            .map(|&id| (id, EntityBundle::new()))
            .collect::<BTreeMap<_, _>>();
        let mut patches = BTreeMap::<EntityId, EntityPatch>::new();
        for &(id, cid) in &self.removed_components {
            patches
                .entry(id)
                .or_default()
                .remove_by_id(cid, "encoded component")?;
        }
        for (run, mut column) in self.column_runs.iter().zip(columns) {
            // Pop in reverse: no shifting, cloning, or reordering entity/value pairs.
            for i in (0..run.entity_len()).rev() {
                let id = run.entity_at(i).unwrap();
                let value = column.swap_remove_erased(column.len() - 1);
                if let Some(bundle) = bundles.get_mut(&id) {
                    bundle.insert_erased(value)?;
                } else {
                    let patch = patches.entry(id).or_default();
                    match run.op_kind {
                        EncodedColumnOpKind::Insert => {
                            patch.insert_erased(value)?;
                        }
                        EncodedColumnOpKind::Replace => {
                            patch.replace_erased(value)?;
                        }
                        EncodedColumnOpKind::Upsert => {
                            patch.upsert_erased(value)?;
                        }
                    }
                }
            }
        }
        let mut batch = StoreBatch::new();
        for (id, bundle) in bundles {
            batch.create_entity(id, bundle)?;
        }
        for &id in &self.deleted_entities {
            batch.delete_entity(id)?;
        }
        for (id, patch) in patches {
            batch.patch_entity(id, patch)?;
        }
        Ok(batch)
    }

    pub(crate) fn decode_values_with_registry(
        &self,
        registry: &ComponentRegistry,
    ) -> Result<Vec<(EntityId, ErasedComponentValue)>, StateError> {
        self.decode_values_with_registry_inner(registry, false)
    }

    pub(crate) fn decode_values_preserving_runs_with_registry(
        &self,
        registry: &ComponentRegistry,
    ) -> Result<Vec<(EntityId, ErasedComponentValue)>, StateError> {
        self.decode_values_with_registry_inner(registry, true)
    }

    fn decode_values_with_registry_inner(
        &self,
        registry: &ComponentRegistry,
        preserve_disjoint_runs: bool,
    ) -> Result<Vec<(EntityId, ErasedComponentValue)>, StateError> {
        self.validate_metadata(registry)?;
        if !self.created_entities.is_empty()
            || !self.deleted_entities.is_empty()
            || !self.removed_components.is_empty()
        {
            return Err(StateError::BatchConflict(
                "Native value replay carries a structural operation".into(),
            ));
        }
        for run in &self.column_runs {
            if !matches!(
                run.op_kind,
                EncodedColumnOpKind::Replace | EncodedColumnOpKind::Upsert
            ) {
                return Err(StateError::BatchConflict(
                    "Native value replay changes component membership".into(),
                ));
            }
        }
        let mut columns = Vec::with_capacity(self.column_runs.len());
        let mut total_values = 0usize;
        for run in &self.column_runs {
            let def = registry
                .get(&run.archived.component_type_id)
                .expect("metadata validated");
            let mut column = def.create_empty_column();
            column.restore_archived_column(&run.archived)?;
            if column.len() != run.entity_len() {
                return Err(StateError::CodecError(
                    "Decoded column row count mismatch".into(),
                ));
            }
            total_values += run.entity_len();
            columns.push(column);
        }
        let mut values = Vec::with_capacity(total_values);
        if self.column_runs.len() > 1 {
            let first_run = &self.column_runs[0];
            let entities_ok = if preserve_disjoint_runs {
                first_run.has_distinct_non_nil_entities()
            } else {
                first_run.is_strictly_increasing_non_nil()
            };
            let uniform_entities = entities_ok
                && self
                    .column_runs
                    .windows(2)
                    .all(|w| w[0].archived.component_type_id < w[1].archived.component_type_id)
                && self.column_runs[1..]
                    .iter()
                    .all(|run| run.same_entities(first_run));
            if uniform_entities {
                let mut decoded_cols: smallvec::SmallVec<[Vec<ErasedComponentValue>; 4]> =
                    smallvec::SmallVec::with_capacity(columns.len());
                for mut column in columns {
                    let n = column.len();
                    let mut col_vals = Vec::with_capacity(n);
                    for _ in 0..n {
                        col_vals.push(column.swap_remove_erased(column.len() - 1));
                    }
                    decoded_cols.push(col_vals);
                }
                for i in 0..first_run.entity_len() {
                    let id = first_run.entity_at(i).unwrap();
                    for col_vals in &mut decoded_cols {
                        values.push((id, col_vals.pop().unwrap()));
                    }
                }
                return Ok(values);
            }
        }
        for (run, mut column) in self.column_runs.iter().zip(columns) {
            let start = values.len();
            for i in (0..run.entity_len()).rev() {
                let id = run.entity_at(i).unwrap();
                let value = column.swap_remove_erased(column.len() - 1);
                values.push((id, value));
            }
            values[start..].reverse();
        }
        if preserve_disjoint_runs
            && self.column_runs.len() == 1
            && self.column_runs[0].has_distinct_non_nil_entities()
        {
            return Ok(values);
        }
        if !values
            .windows(2)
            .all(|w| (w[0].0, w[0].1.component_type_id()) < (w[1].0, w[1].1.component_type_id()))
        {
            values.sort_unstable_by_key(|(id, val)| (*id, val.component_type_id()));
        }
        Ok(values)
    }
}

impl StoreBatch {
    /// Schemas/codec versions come from the registered capability, never from
    /// the erased value's serde codec or a second ad-hoc registry.
    pub(crate) fn encode_with_registry(
        &self,
        registry: &ComponentRegistry,
    ) -> Result<EncodedStoreBatch, StateError> {
        type Rows<'a> = (Vec<EntityId>, Vec<&'a ErasedComponentValue>);
        let mut groups = BTreeMap::<(ComponentTypeId, EncodedColumnOpKind), Rows<'_>>::new();
        let mut encoded = EncodedStoreBatch::new();
        for (&id, op) in &self.ops {
            let mut add = |cid, kind, value| {
                let (ids, values) = groups.entry((cid, kind)).or_default();
                ids.push(id);
                values.push(value);
            };
            match op {
                BatchOp::CreateEntity { bundle, .. } => {
                    encoded.created_entities.push(id);
                    for (&cid, value) in bundle.components() {
                        add(cid, EncodedColumnOpKind::Insert, value);
                    }
                }
                BatchOp::DeleteEntity { .. } => encoded.deleted_entities.push(id),
                BatchOp::PatchEntity { patch, .. } => patch.for_each_op(|cid, op| match op {
                    PatchOp::Insert(value) => add(cid, EncodedColumnOpKind::Insert, value),
                    PatchOp::Replace(value) => add(cid, EncodedColumnOpKind::Replace, value),
                    PatchOp::Upsert(value) => add(cid, EncodedColumnOpKind::Upsert, value),
                    PatchOp::Remove(_) => encoded.removed_components.push((id, cid)),
                }),
            }
        }
        for ((cid, op_kind), (entities, values)) in groups {
            let def = registry
                .get(&cid)
                .ok_or_else(|| StateError::TypeNotRegistered(format!("{cid:?}")))?;
            let archived = def.prototype().archive_erased_values(&values)?;
            encoded.column_runs.push(EncodedColumnRun {
                entities,
                shared_entities: None,
                op_kind,
                archived,
            });
        }
        encoded.removed_components.sort_unstable();
        // Same wire invariants for native encoding and untrusted decoding.
        encoded.validate_metadata(registry)?;
        Ok(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::Options;
    use uuid::Uuid;

    fn dummy_run(entities: Vec<EntityId>) -> EncodedColumnRun {
        let row_count = entities.len() as u32;
        EncodedColumnRun {
            entities,
            shared_entities: None,
            op_kind: EncodedColumnOpKind::Replace,
            archived: ArchivedColumnData {
                component_type_id: ComponentTypeId::new(1),
                schema_version: 1,
                codec: ColumnCodecId::Rkyv,
                row_count,
                payload: Vec::new(),
            },
        }
    }

    fn bincode_roundtrip(run: &EncodedColumnRun) -> (Vec<u8>, EncodedColumnRun) {
        let opts = bincode::DefaultOptions::new().with_fixint_encoding();
        let encoded = opts.serialize(run).expect("bincode serialize should succeed");
        let decoded: EncodedColumnRun = opts
            .deserialize(&encoded)
            .expect("bincode deserialize should succeed");
        assert_eq!(run, &decoded);
        (encoded, decoded)
    }

    #[test]
    fn delta_rle_roundtrip_empty_and_small() {
        for count in 0..4 {
            let entities: Vec<EntityId> = (0..count)
                .map(|i| EntityId::from_u128(1000 + i as u128))
                .collect();
            let run = dummy_run(entities);
            let (_bytes, decoded) = bincode_roundtrip(&run);
            assert_eq!(decoded.entities.len(), count);
        }
    }

    #[test]
    fn delta_rle_roundtrip_50k_contiguous_and_strided() {
        let entities: Vec<EntityId> = (0..50_000u128)
            .map(|i| EntityId::from_u128(0x0190_0000_0000_0000_0000_0000_0000_0001 + i * 7))
            .collect();
        let compressed = entity_slice_serde::encode_delta_rle(&entities)
            .expect("contiguous/strided 50k entities should compress");
        assert!(
            compressed.len() < 64,
            "expected < 64 bytes for 50k strided entities, got {}",
            compressed.len()
        );
        assert_ne!(compressed.len() % 16, 0);

        let run = dummy_run(entities);
        assert_eq!(run.estimated_entities_wire_bytes(), 64);
        let (wire_bytes, decoded) = bincode_roundtrip(&run);
        assert!(wire_bytes.len() < 128);
        assert_eq!(decoded.entities.len(), 50_000);
    }

    #[test]
    fn delta_rle_roundtrip_piecewise_multi_range_with_holes() {
        let mut entities = Vec::new();
        for seg in 0..15u128 {
            let high = if seg >= 10 { 1u128 << 65 } else if seg >= 5 { 1u128 << 64 } else { 0 };
            let base = high + seg * 10_000 + 1;
            for i in 0..100u128 {
                entities.push(EntityId::from_u128(base + i));
            }
        }
        let compressed = entity_slice_serde::encode_delta_rle(&entities)
            .expect("piecewise multi-range entities should compress");
        assert!(
            compressed.len() < 500,
            "expected < 500 bytes for piecewise multi-range, got {}",
            compressed.len()
        );
        assert_ne!(compressed.len() % 16, 0);

        let run = dummy_run(entities);
        let (_wire_bytes, _decoded) = bincode_roundtrip(&run);
    }

    #[test]
    fn delta_rle_roundtrip_sparse_increasing_same_upper_u64() {
        let mut cur = 0xabcd_ef01_2345_6789_0000_0000_0000_0010u128;
        let mut entities = Vec::with_capacity(200);
        for i in 0..200u128 {
            cur += (i * 37 + 11) % 997 + 1;
            entities.push(EntityId::from_u128(cur));
        }
        let compressed = entity_slice_serde::encode_delta_rle(&entities)
            .expect("sparse u64-delta entities should compress");
        assert!(
            compressed.len() < entities.len() * 10,
            "expected narrow u64 delta compression, got {} vs raw {}",
            compressed.len(),
            entities.len() * 16
        );
        assert_ne!(compressed.len() % 16, 0);

        let run = dummy_run(entities);
        let (_wire_bytes, _decoded) = bincode_roundtrip(&run);
    }

    #[test]
    fn delta_rle_fallback_random_uuid_v4_no_bloat() {
        let entities: Vec<EntityId> = (0..1_000)
            .map(|_| EntityId::from_uuid(Uuid::new_v4()))
            .collect();
        assert!(
            entity_slice_serde::encode_delta_rle(&entities).is_none(),
            "random UUIDv4 should fallback to raw 16*N bytes without bloat"
        );
        let run = dummy_run(entities);
        let (wire_bytes, decoded) = bincode_roundtrip(&run);
        assert_eq!(decoded.entities.len(), 1_000);
        // Raw 1000 * 16 = 16000 bytes + small struct framing overhead (~40 bytes)
        assert!(wire_bytes.len() < 16_064);
    }

    #[test]
    fn delta_rle_u128_runs_u128_deltas_and_16_byte_padding_boundary() {
        // 1. Single-stride TAG_RUN_U128 (constant stride > u64::MAX across 8 entities)
        let stride_u128 = (1u128 << 64) + 13;
        let single_u128_run: Vec<EntityId> = (0..8u128)
            .map(|i| EntityId::from_u128(100 + i * stride_u128))
            .collect();
        let compressed_single = entity_slice_serde::encode_delta_rle(&single_u128_run)
            .expect("single u128 stride should compress via TAG_RUN_U128");
        // 1 (magic) + 4 (count) + 16 (first) + 1 (TAG_RUN_U128) + 4 (run_len) + 16 (stride) = 42 bytes
        assert_eq!(compressed_single.len(), 42);
        let (_, decoded_single) = bincode_roundtrip(&dummy_run(single_u128_run));
        assert_eq!(decoded_single.entities.len(), 8);

        // 2. Piecewise TAG_DELTAS_U128 (non-uniform deltas > u64::MAX) + TAG_RUN_U128
        let mut mixed_u128 = Vec::new();
        let mut cur = 1_000u128;
        mixed_u128.push(EntityId::from_u128(cur));
        // 3 distinct wide u128 deltas -> emitted via TAG_DELTAS_U128
        for d in [(1u128 << 64) + 7, (1u128 << 65) + 19, (1u128 << 64) + 43] {
            cur = cur.wrapping_add(d);
            mixed_u128.push(EntityId::from_u128(cur));
        }
        // Followed by a 10-step uniform wide u128 run -> emitted via TAG_RUN_U128
        for _ in 0..10 {
            cur = cur.wrapping_add((1u128 << 64) + 99);
            mixed_u128.push(EntityId::from_u128(cur));
        }
        let compressed_mixed = entity_slice_serde::encode_delta_rle(&mixed_u128)
            .expect("mixed TAG_DELTAS_U128 + TAG_RUN_U128 should compress");
        assert!(compressed_mixed.len() < mixed_u128.len() * 16);
        let (_, decoded_mixed) = bincode_roundtrip(&dummy_run(mixed_u128));
        assert_eq!(decoded_mixed.entities.len(), 14);

        // 3. Exact 16-byte-multiple pre-padding length (112 bytes -> padded with 1 trailing 0u8 byte to 113 bytes):
        // Header = 21 bytes; 7 segments of TAG_RUN_U64 (13 bytes each) = 91 bytes; 21 + 91 = 112 bytes (112 % 16 == 0).
        let mut padded_entities = Vec::with_capacity(15);
        let mut pos = 500u128;
        padded_entities.push(EntityId::from_u128(pos));
        for stride in 1..=7u128 {
            for _ in 0..2 {
                pos += stride;
                padded_entities.push(EntityId::from_u128(pos));
            }
        }
        assert_eq!(padded_entities.len(), 15);
        let compressed_padded = entity_slice_serde::encode_delta_rle(&padded_entities)
            .expect("7-segment TAG_RUN_U64 should compress and trigger 1-byte padding");
        assert_eq!(compressed_padded.len(), 113);
        assert_eq!(*compressed_padded.last().unwrap(), 0u8);
        let (_, decoded_padded) = bincode_roundtrip(&dummy_run(padded_entities));
        assert_eq!(decoded_padded.entities.len(), 15);
    }

    #[test]
    fn test_encoded_column_run_interleaved_decode_values_with_registry_fallback_sort() {
        let mut store = crate::Store::new();
        store.register_component::<u64>().unwrap();
        let cid = ComponentTypeId::of::<u64>();
        let archive = store
            .component_registry
            .get(&cid)
            .unwrap()
            .prototype()
            .raw_archive_capability()
            .unwrap();

        let encode_u64_slice = |vals: &[u64]| -> ArchivedColumnData {
            let slice = crate::storage::core::column::archive::RawArchiveSlice {
                base: vals.as_ptr() as *const u8,
                offset: 0,
                len: vals.len(),
            };
            unsafe { (archive.encode)(archive.component, archive.schema_version, &[slice]).unwrap() }
        };

        // Two runs for the same component with interleaved entity IDs: [10, 30] and [20, 40]
        let mut batch = EncodedStoreBatch::new();
        batch.column_runs.push(EncodedColumnRun {
            entities: vec![EntityId::from_u128(10), EntityId::from_u128(30)],
            shared_entities: None,
            op_kind: EncodedColumnOpKind::Replace,
            archived: encode_u64_slice(&[100u64, 300u64]),
        });
        batch.column_runs.push(EncodedColumnRun {
            entities: vec![EntityId::from_u128(20), EntityId::from_u128(40)],
            shared_entities: None,
            op_kind: EncodedColumnOpKind::Replace,
            archived: encode_u64_slice(&[200u64, 400u64]),
        });

        let decoded = batch
            .decode_values_with_registry(&store.component_registry)
            .unwrap();
        let pairs: Vec<(EntityId, u64)> = decoded
            .iter()
            .map(|(id, val)| (*id, *val.downcast_ref::<u64>().unwrap()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                (EntityId::from_u128(10), 100),
                (EntityId::from_u128(20), 200),
                (EntityId::from_u128(30), 300),
                (EntityId::from_u128(40), 400),
            ]
        );
    }
}

