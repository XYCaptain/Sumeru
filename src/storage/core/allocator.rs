//! Stable entity identities mapped to store-private, generational slots.

use crate::binding::EntityId;
use crate::error::StateError;
use crate::id::LocalSlotId;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;
use super::entity_index::EntityIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct EntityLocation {
    pub archetype_id: u32,
    pub row_index: u32,
}

impl EntityLocation {
    pub const fn new(archetype_id: u32, row_index: u32) -> Self {
        Self {
            archetype_id,
            row_index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SlotMeta {
    pub(crate) generation: u32,
    pub(crate) entity: Option<EntityId>,
    pub(crate) location: Option<EntityLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ActiveSlotRun {
    start_slot: u32,
    generation: u32,
    archetype_id: u32,
    base_row_index: u32,
    #[serde(with = "crate::storage::core::entity_index::entity_slice_serde")]
    entities: Vec<EntityId>,
}

mod slots_serde {
    use super::{ActiveSlotRun, EntityLocation, SlotMeta};
    use crate::binding::EntityId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    const COMPACT_SLOTS_TAG: u64 = 1u64 << 63;

    struct SlotRunEntities<'a>(&'a [SlotMeta]);

    impl Serialize for SlotRunEntities<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if !serializer.is_human_readable() && self.0.len() >= 4 {
                let first = self.0[0].entity.unwrap().to_u128();
                let second = self.0[1].entity.unwrap().to_u128();
                let stride = second.wrapping_sub(first);
                let is_single_run = self.0[2..].iter().enumerate().all(|(i, m)| {
                    m.entity.unwrap().to_u128()
                        == first.wrapping_add(stride.wrapping_mul((i + 2) as u128))
                });
                if is_single_run {
                    if let Some(compressed) =
                        crate::storage::core::entity_index::entity_slice_serde::encode_constant_stride_run(
                            self.0.len(),
                            first,
                            stride,
                        )
                    {
                        return serializer.serialize_bytes(&compressed);
                    }
                }
            }
            let entities: Vec<EntityId> = self.0.iter().map(|m| m.entity.unwrap()).collect();
            crate::storage::core::entity_index::entity_slice_serde::serialize(&entities, serializer)
        }
    }

    #[derive(Serialize)]
    struct ActiveSlotRunRef<'a> {
        start_slot: u32,
        generation: u32,
        archetype_id: u32,
        base_row_index: u32,
        entities: SlotRunEntities<'a>,
    }

    #[inline]
    fn active_run_len_from(slots: &[SlotMeta], start: usize) -> usize {
        let first = &slots[start];
        let (Some(_), Some(loc0)) = (first.entity, first.location) else {
            return 0;
        };
        let gen0 = first.generation;
        let arch0 = loc0.archetype_id;
        let row0 = loc0.row_index;
        let mut len = 1usize;
        for item in &slots[start + 1..] {
            let (Some(_), Some(loc)) = (item.entity, item.location) else {
                break;
            };
            let Ok(offset) = u32::try_from(len) else {
                break;
            };
            let Some(expected_row) = row0.checked_add(offset) else {
                break;
            };
            if item.generation != gen0
                || loc.archetype_id != arch0
                || loc.row_index != expected_row
            {
                break;
            }
            len += 1;
        }
        len
    }

    pub(super) fn serialize<S: Serializer>(
        slots: &[SlotMeta],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() || slots.len() < 4 {
            return slots.serialize(serializer);
        }
        let total_slots = u32::try_from(slots.len())
            .map_err(|_| serde::ser::Error::custom("EntityAllocator slots length exceeds u32"))?;

        let mut runs: Vec<ActiveSlotRunRef<'_>> = Vec::new();
        let mut literals: Vec<(u32, SlotMeta)> = Vec::new();
        let mut idx = 0usize;
        while idx < slots.len() {
            let run_len = active_run_len_from(slots, idx);
            if run_len >= 4 {
                let first = &slots[idx];
                let loc0 = first.location.unwrap();
                runs.push(ActiveSlotRunRef {
                    start_slot: idx as u32,
                    generation: first.generation,
                    archetype_id: loc0.archetype_id,
                    base_row_index: loc0.row_index,
                    entities: SlotRunEntities(&slots[idx..idx + run_len]),
                });
                idx += run_len;
            } else {
                let step = run_len.max(1);
                for offset in 0..step {
                    let s = idx + offset;
                    literals.push((s as u32, slots[s]));
                }
                idx += step;
            }
        }

        use serde::ser::SerializeTuple;
        let header = COMPACT_SLOTS_TAG | (total_slots as u64);
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&header)?;
        tup.serialize_element(&runs)?;
        tup.serialize_element(&literals)?;
        tup.end()
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<SlotMeta>, D::Error> {
        if deserializer.is_human_readable() {
            return Vec::<SlotMeta>::deserialize(deserializer);
        }

        struct SlotsVisitor;
        impl<'de> serde::de::Visitor<'de> for SlotsVisitor {
            type Value = Vec<SlotMeta>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a sequence of SlotMeta or compact ActiveSlotRun + literals")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                use serde::de::Error;
                let header: u64 = seq
                    .next_element()?
                    .ok_or_else(|| Error::custom("Missing EntityAllocator slots header"))?;

                if (header & COMPACT_SLOTS_TAG) == 0 {
                    if header >= u32::MAX as u64 {
                        return Err(Error::custom("EntityAllocator slots length exceeds u32"));
                    }
                    let count = header as usize;
                    let mut out = Vec::<SlotMeta>::new();
                    out.try_reserve_exact(count.min(65536))
                        .map_err(|_| Error::custom("EntityAllocator slots allocation failed"))?;
                    for _ in 0..count {
                        let meta: SlotMeta = seq
                            .next_element()?
                            .ok_or_else(|| Error::custom("Truncated EntityAllocator slots"))?;
                        out.push(meta);
                    }
                    return Ok(out);
                }

                let raw_total = header & !COMPACT_SLOTS_TAG;
                if raw_total >= u32::MAX as u64 {
                    return Err(Error::custom(
                        "EntityAllocator compact total_slots exceeds u32",
                    ));
                }
                let total_slots = raw_total as usize;
                let runs: Vec<ActiveSlotRun> = seq
                    .next_element()?
                    .ok_or_else(|| Error::custom("Missing EntityAllocator compact runs"))?;
                let literals: Vec<(u32, SlotMeta)> = seq
                    .next_element()?
                    .ok_or_else(|| Error::custom("Missing EntityAllocator compact literals"))?;

                let mut covered = literals.len();
                for run in &runs {
                    covered = covered
                        .checked_add(run.entities.len())
                        .ok_or_else(|| {
                            Error::custom("EntityAllocator compact slot count overflow")
                        })?;
                }
                if covered != total_slots {
                    return Err(Error::custom(
                        "EntityAllocator compact runs and literals count mismatch",
                    ));
                }

                let mut slots = Vec::<SlotMeta>::new();
                slots
                    .try_reserve_exact(total_slots)
                    .map_err(|_| Error::custom("EntityAllocator slots allocation failed"))?;
                slots.resize(
                    total_slots,
                    SlotMeta {
                        generation: 0,
                        entity: None,
                        location: None,
                    },
                );

                let words = total_slots.div_ceil(64);
                let mut populated = Vec::<u64>::new();
                populated
                    .try_reserve_exact(words)
                    .map_err(|_| Error::custom("EntityAllocator bitmap allocation failed"))?;
                populated.resize(words, 0);

                for run in runs {
                    let run_len = run.entities.len();
                    let start = run.start_slot as usize;
                    let end = start
                        .checked_add(run_len)
                        .ok_or_else(|| Error::custom("ActiveSlotRun slot range overflow"))?;
                    if end > total_slots {
                        return Err(Error::custom("ActiveSlotRun exceeds total_slots"));
                    }
                    if run_len > 0 {
                        let last_offset = u32::try_from(run_len - 1)
                            .map_err(|_| Error::custom("ActiveSlotRun length overflow"))?;
                        if run.base_row_index.checked_add(last_offset).is_none() {
                            return Err(Error::custom("ActiveSlotRun row_index overflow"));
                        }
                    }
                    for (offset, entity) in run.entities.into_iter().enumerate() {
                        let slot_idx = start + offset;
                        let word = &mut populated[slot_idx >> 6];
                        let mask = 1u64 << (slot_idx & 63);
                        if (*word & mask) != 0 {
                            return Err(Error::custom(
                                "Duplicate slot index in EntityAllocator compact runs",
                            ));
                        }
                        *word |= mask;
                        slots[slot_idx] = SlotMeta {
                            generation: run.generation,
                            entity: Some(entity),
                            location: Some(EntityLocation::new(
                                run.archetype_id,
                                run.base_row_index + offset as u32,
                            )),
                        };
                    }
                }

                for (slot_u32, meta) in literals {
                    let slot_idx = slot_u32 as usize;
                    if slot_idx >= total_slots {
                        return Err(Error::custom(
                            "Literal slot index out of bounds in EntityAllocator",
                        ));
                    }
                    let word = &mut populated[slot_idx >> 6];
                    let mask = 1u64 << (slot_idx & 63);
                    if (*word & mask) != 0 {
                        return Err(Error::custom(
                            "Duplicate literal slot index in EntityAllocator",
                        ));
                    }
                    *word |= mask;
                    slots[slot_idx] = meta;
                }

                Ok(slots)
            }
        }

        deserializer.deserialize_tuple(usize::MAX, SlotsVisitor)
    }
}

/// Stores never manufacture business identities. Two independent stores may
/// install the same EntityId at different slots and generations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub(crate) struct EntityAllocator {
    #[serde(with = "slots_serde")]
    slots: Vec<SlotMeta>,
    free_slots: Vec<u32>,
    /// Rebuilt from authoritative slot metadata when decoding a snapshot.
    #[serde(skip)]
    by_entity: EntityIndex<LocalSlotId>,
}

impl<'de> Deserialize<'de> for EntityAllocator {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Image {
            #[serde(with = "slots_serde")]
            slots: Vec<SlotMeta>,
            free_slots: Vec<u32>,
        }
        let image = Image::deserialize(deserializer)?;
        let mut allocator = Self {
            slots: image.slots,
            free_slots: image.free_slots,
            by_entity: EntityIndex::default(),
        };
        allocator.by_entity = allocator
            .rebuild_index()
            .map_err(serde::de::Error::custom)?;
        Ok(allocator)
    }
}

#[allow(dead_code)]
impl EntityAllocator {
    /// Snapshot codec primitives; derived hash indices are never persisted.
    pub(crate) fn snapshot_slots(&self) -> impl Iterator<Item = (u32, Option<EntityId>, Option<(u32, u32)>)> + '_ {
        self.slots.iter().map(|slot| (slot.generation, slot.entity,
            slot.location.map(|loc| (loc.archetype_id, loc.row_index))))
    }

    pub(crate) fn snapshot_free_slots(&self) -> &[u32] { &self.free_slots }

    pub(crate) fn from_snapshot_parts(
        slots: Vec<(u32, Option<EntityId>, Option<(u32, u32)>)>, free_slots: Vec<u32>,
    ) -> Result<Self, StateError> {
        let mut allocator = Self {
            slots: slots.into_iter().map(|(generation, entity, location)| SlotMeta {
                generation, entity,
                location: location.map(|(arch, row)| EntityLocation::new(arch, row)),
            }).collect(),
            free_slots,
            by_entity: EntityIndex::default(),
        };
        allocator.by_entity = allocator.rebuild_index()?;
        allocator.validate()?;
        Ok(allocator)
    }


    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free_slots.clear();
        self.by_entity.clear();
    }

    #[inline(always)]
    pub(crate) fn slot_generation(&self, slot: u32) -> Option<u32> {
        self.slots.get(slot as usize).map(|m| m.generation)
    }

    #[inline(always)]
    pub(crate) fn has_no_custom_identities(&self) -> bool {
        self.by_entity.is_empty()
    }

    /// Validate capacity and the free slots a batch may consume before any row write.
    pub(crate) fn prepare_allocations(&mut self, count: usize) -> Result<(), StateError> {
        let reused = count.min(self.free_slots.len());
        for &slot in self.free_slots.iter().rev().take(reused) {
            let meta = self
                .slots
                .get(slot as usize)
                .ok_or_else(|| StateError::AccessDenied("Invalid free slot".into()))?;
            if meta.entity.is_some() || meta.location.is_some() || meta.generation == 0 {
                return Err(StateError::AccessDenied(
                    "Invalid free slot metadata".into(),
                ));
            }
        }
        let additional = count - reused;
        let size = self
            .slots
            .len()
            .checked_add(additional)
            .ok_or_else(|| StateError::CapacityExhausted("Entity slots exhausted".into()))?;
        if size >= u32::MAX as usize {
            return Err(StateError::CapacityExhausted(
                "Entity slots exhausted".into(),
            ));
        }
        self.slots
            .try_reserve(additional)
            .map_err(|_| StateError::CapacityExhausted("Entity slot reservation failed".into()))?;
        if !self.by_entity.is_empty() {
            self.by_entity
                .try_reserve(count)
                .map_err(|_| StateError::CapacityExhausted("Entity hash index reservation failed".into()))?;
        }
        Ok(())
    }

    pub(crate) fn validate_allocate(&self, entity: EntityId) -> Result<(), StateError> {
        if entity.is_nil() {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }
        if self.is_alive(entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} is already installed"
            )));
        }
        Ok(())
    }

    pub fn allocate(&mut self, entity: EntityId) -> Result<LocalSlotId, StateError> {
        if self.free_slots.is_empty() {
            let slot = u32::try_from(self.slots.len()).map_err(|_| {
                StateError::CapacityExhausted("Entity slot capacity exceeds u32".into())
            })?;
            if slot == u32::MAX {
                return Err(StateError::CapacityExhausted(
                    "Entity slot capacity u32::MAX reached".into(),
                ));
            }
            if Self::is_identity_slot(entity, slot) {
                if !self.by_entity.is_empty() && self.by_entity.contains_key(&entity) {
                    return Err(StateError::AccessDenied(format!(
                        "Entity {entity} is already installed"
                    )));
                }
                let local = LocalSlotId::new(slot, 1);
                self.slots.push(SlotMeta {
                    generation: 1,
                    entity: Some(entity),
                    location: None,
                });
                return Ok(local);
            }
            if entity.is_nil() {
                return Err(StateError::AccessDenied(
                    "Cannot install a nil entity identity".into(),
                ));
            }
            if self.candidate_slot_meta(entity).is_some() {
                return Err(StateError::AccessDenied(format!(
                    "Entity {entity} is already installed"
                )));
            }
            let local = LocalSlotId::new(slot, 1);
            match self.by_entity.entry(entity) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(StateError::AccessDenied(format!(
                        "Entity {entity} is already installed"
                    )));
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(local);
                }
            }
            self.slots.push(SlotMeta {
                generation: 1,
                entity: Some(entity),
                location: None,
            });
            return Ok(local);
        }
        if entity.is_nil() {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }
        if self.is_alive(entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} is already installed"
            )));
        }
        let local = if let Some(&slot) = self.free_slots.last() {
            let meta = self.slots.get(slot as usize).ok_or_else(|| {
                StateError::AccessDenied(format!("Free slot {slot} is out of bounds"))
            })?;
            if meta.entity.is_some() || meta.location.is_some() || meta.generation == 0 {
                return Err(StateError::AccessDenied(format!(
                    "Free slot {slot} has invalid metadata"
                )));
            }
            LocalSlotId::new(slot, meta.generation)
        } else {
            unreachable!()
        };
        self.free_slots.pop();
        self.slots[local.slot as usize].entity = Some(entity);
        if !Self::is_identity_slot(entity, local.slot) {
            self.by_entity.insert(entity, local);
        }
        Ok(local)
    }

    #[inline]
    pub(crate) fn allocate_with_location(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
    ) -> Result<LocalSlotId, StateError> {
        let local = self.allocate(entity)?;
        self.slots[local.slot as usize].location = Some(location);
        Ok(local)
    }

    /// 单次哈希探测分配并绑定物理位置（兼顾全新槽位与空闲槽位复用，零重复查询）。
    /// 若实体已存活返回 `EntityAlreadyExists(entity)`。
    #[inline]
    pub(crate) fn allocate_unique_with_location(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
    ) -> Result<LocalSlotId, StateError> {
        if self.free_slots.is_empty() {
            let slot = u32::try_from(self.slots.len()).map_err(|_| {
                StateError::CapacityExhausted("Entity slots exhausted".into())
            })?;
            if slot == u32::MAX {
                return Err(StateError::CapacityExhausted(
                    "Entity slots exhausted".into(),
                ));
            }
            if Self::is_identity_slot(entity, slot) {
                if !self.by_entity.is_empty() {
                    if let Some(&local) = self.by_entity.get(&entity) {
                        if self
                            .slots
                            .get(local.slot as usize)
                            .is_some_and(|meta| {
                                meta.entity == Some(entity) && meta.generation == local.generation
                            })
                        {
                            return Err(StateError::EntityAlreadyExists(entity));
                        }
                        return Err(StateError::AccessDenied(format!(
                            "Entity {entity} is already installed"
                        )));
                    }
                }
                let local = LocalSlotId::new(slot, 1);
                self.slots.push(SlotMeta {
                    generation: 1,
                    entity: Some(entity),
                    location: Some(location),
                });
                return Ok(local);
            }
        }
        if entity.is_nil() {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }
        if self.candidate_slot_meta(entity).is_some() {
            return Err(StateError::EntityAlreadyExists(entity));
        }
        if let Some(&slot) = self.free_slots.last() {
            if Self::is_identity_slot(entity, slot) {
                if !self.by_entity.is_empty() {
                    if let Some(&local) = self.by_entity.get(&entity) {
                        if self
                            .slots
                            .get(local.slot as usize)
                            .is_some_and(|meta| {
                                meta.entity == Some(entity) && meta.generation == local.generation
                            })
                        {
                            return Err(StateError::EntityAlreadyExists(entity));
                        }
                        return Err(StateError::AccessDenied(format!(
                            "Entity {entity} is already installed"
                        )));
                    }
                }
                let meta = self
                    .slots
                    .get_mut(slot as usize)
                    .ok_or_else(|| StateError::AccessDenied("Invalid free slot".into()))?;
                if meta.entity.is_some() || meta.location.is_some() || meta.generation == 0 {
                    return Err(StateError::AccessDenied(
                        "Invalid free slot metadata".into(),
                    ));
                }
                let local = LocalSlotId::new(slot, meta.generation);
                meta.entity = Some(entity);
                meta.location = Some(location);
                self.free_slots.pop();
                return Ok(local);
            }
        }
        match self.by_entity.entry(entity) {
            std::collections::hash_map::Entry::Occupied(occ) => {
                let local = *occ.get();
                if self
                    .slots
                    .get(local.slot as usize)
                    .is_some_and(|meta| meta.entity == Some(entity) && meta.generation == local.generation)
                {
                    return Err(StateError::EntityAlreadyExists(entity));
                }
                Err(StateError::AccessDenied(format!(
                    "Entity {entity} is already installed"
                )))
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                if let Some(&slot) = self.free_slots.last() {
                    let meta = self
                        .slots
                        .get_mut(slot as usize)
                        .ok_or_else(|| StateError::AccessDenied("Invalid free slot".into()))?;
                    if meta.entity.is_some() || meta.location.is_some() || meta.generation == 0 {
                        return Err(StateError::AccessDenied(
                            "Invalid free slot metadata".into(),
                        ));
                    }
                    let local = LocalSlotId::new(slot, meta.generation);
                    meta.entity = Some(entity);
                    meta.location = Some(location);
                    self.free_slots.pop();
                    v.insert(local);
                    Ok(local)
                } else {
                    let slot = u32::try_from(self.slots.len()).map_err(|_| {
                        StateError::CapacityExhausted("Entity slots exhausted".into())
                    })?;
                    if slot == u32::MAX {
                        return Err(StateError::CapacityExhausted(
                            "Entity slots exhausted".into(),
                        ));
                    }
                    let local = LocalSlotId::new(slot, 1);
                    self.slots.push(SlotMeta {
                        generation: 1,
                        entity: Some(entity),
                        location: Some(location),
                    });
                    v.insert(local);
                    Ok(local)
                }
            }
        }
    }

    /// Fast path for fresh `Store` construction (`free_slots` is empty): validates
    /// non-nil and uniqueness while allocating the slot and binding its physical
    /// `EntityLocation` in a single `EntityIndex::insert` without redundant lookups.
    #[inline]
    pub(crate) fn allocate_fresh_with_location(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
    ) -> Result<LocalSlotId, StateError> {
        if entity.is_nil() {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }
        debug_assert!(self.free_slots.is_empty());
        let slot = u32::try_from(self.slots.len()).map_err(|_| {
            StateError::CapacityExhausted("Entity slot capacity exceeds u32".into())
        })?;
        if slot == u32::MAX {
            return Err(StateError::CapacityExhausted(
                "Entity slot capacity u32::MAX reached".into(),
            ));
        }
        if Self::is_identity_slot(entity, slot) {
            if !self.by_entity.is_empty() && self.by_entity.contains_key(&entity) {
                return Err(StateError::EntityAlreadyExists(entity));
            }
            let local = LocalSlotId::new(slot, 1);
            self.slots.push(SlotMeta {
                generation: 1,
                entity: Some(entity),
                location: Some(location),
            });
            return Ok(local);
        }
        if self.candidate_slot_meta(entity).is_some() {
            return Err(StateError::EntityAlreadyExists(entity));
        }
        let local = LocalSlotId::new(slot, 1);
        match self.by_entity.entry(entity) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(StateError::EntityAlreadyExists(entity));
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(local);
            }
        }
        self.slots.push(SlotMeta {
            generation: 1,
            entity: Some(entity),
            location: Some(location),
        });
        Ok(local)
    }

    /// 批量分配具有连续行号的新实体槽位（供多分区快照恢复 Store 合并使用）
    pub(crate) fn allocate_batch_with_contiguous_rows(
        &mut self,
        entities: &[EntityId],
        archetype_id: u32,
        base_row: u32,
    ) -> Result<(), StateError> {
        if entities.is_empty() {
            return Ok(());
        }
        let last_offset = u32::try_from(entities.len() - 1)
            .map_err(|_| StateError::CapacityExhausted("Archetype row capacity exceeds u32".into()))?;
        if base_row.checked_add(last_offset).is_none() {
            return Err(StateError::CapacityExhausted(
                "Archetype row capacity exceeds u32".into(),
            ));
        }
        if entities.iter().any(|e| e.is_nil()) {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }
        self.prepare_allocations(entities.len())?;
        if self.free_slots.is_empty() {
            let base_slot = self.slots.len() as u32;
            for (offset, &entity) in entities.iter().enumerate() {
                let slot = base_slot + offset as u32;
                let local = LocalSlotId::new(slot, 1);
                if Self::is_identity_slot(entity, slot) {
                    if !self.by_entity.is_empty() && self.by_entity.contains_key(&entity) {
                        for (prev_off, &prev) in entities[..offset].iter().enumerate() {
                            if !Self::is_identity_slot(prev, base_slot + prev_off as u32) {
                                self.by_entity.remove(&prev);
                            }
                        }
                        self.slots.truncate(base_slot as usize);
                        return Err(StateError::EntityAlreadyExists(entity));
                    }
                } else {
                    if self.candidate_slot_meta(entity).is_some() {
                        for (prev_off, &prev) in entities[..offset].iter().enumerate() {
                            if !Self::is_identity_slot(prev, base_slot + prev_off as u32) {
                                self.by_entity.remove(&prev);
                            }
                        }
                        self.slots.truncate(base_slot as usize);
                        return Err(StateError::EntityAlreadyExists(entity));
                    }
                    match self.by_entity.entry(entity) {
                        std::collections::hash_map::Entry::Occupied(_) => {
                            for (prev_off, &prev) in entities[..offset].iter().enumerate() {
                                if !Self::is_identity_slot(prev, base_slot + prev_off as u32) {
                                    self.by_entity.remove(&prev);
                                }
                            }
                            self.slots.truncate(base_slot as usize);
                            return Err(StateError::EntityAlreadyExists(entity));
                        }
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(local);
                        }
                    }
                }
                self.slots.push(SlotMeta {
                    generation: 1,
                    entity: Some(entity),
                    location: Some(EntityLocation::new(archetype_id, base_row + offset as u32)),
                });
            }
            return Ok(());
        }
        for (i, &entity) in entities.iter().enumerate() {
            if self.is_alive(entity) || entities[..i].contains(&entity) {
                return Err(StateError::EntityAlreadyExists(entity));
            }
        }
        for (offset, &entity) in entities.iter().enumerate() {
            let local = self.allocate(entity)?;
            self.slots[local.slot as usize].location =
                Some(EntityLocation::new(archetype_id, base_row + offset as u32));
        }
        Ok(())
    }

    pub fn deallocate(&mut self, entity: EntityId) -> Result<Option<EntityLocation>, StateError> {
        self.validate_deallocate(entity)?;
        Ok(self.deallocate_infallible(entity))
    }

    /// Check identity and generation exhaustion without changing allocator state.
    pub(crate) fn validate_deallocate(&self, entity: EntityId) -> Result<(), StateError> {
        let local = self
            .local_slot(entity)
            .ok_or_else(|| StateError::AccessDenied(format!("Entity {entity} is not installed")))?;
        self.slots[local.slot as usize]
            .generation
            .checked_add(1)
            .ok_or(StateError::GenerationOverflow)?;
        Ok(())
    }

    /// Requires successful `validate_deallocate` with no intervening change to
    /// this entity's allocation. Other validated entities may be deallocated.
    pub(crate) fn deallocate_infallible(&mut self, entity: EntityId) -> Option<EntityLocation> {
        let local = self
            .local_slot(entity)
            .expect("validated entity must remain installed");
        self.deallocate_slot_infallible(entity, local)
    }

    #[inline(always)]
    fn is_identity_slot(entity: EntityId, slot: u32) -> bool {
        !entity.is_nil() && (entity.to_u128() as u32).wrapping_sub(1) == slot
    }

    #[inline(always)]
    fn candidate_slot_meta(&self, entity: EntityId) -> Option<(u32, &SlotMeta)> {
        let idx = (entity.to_u128() as u32).wrapping_sub(1);
        let meta = self.slots.get(idx as usize)?;
        if meta.entity == Some(entity) {
            Some((idx, meta))
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn local_slot_and_location(
        &self,
        entity: EntityId,
    ) -> Option<(LocalSlotId, EntityLocation)> {
        if let Some((slot, meta)) = self.candidate_slot_meta(entity) {
            let loc = meta.location?;
            return Some((LocalSlotId::new(slot, meta.generation), loc));
        }
        if self.by_entity.is_empty() {
            return None;
        }
        let local = *self.by_entity.get(&entity)?;
        let loc = self.slots.get(local.slot as usize)?.location?;
        Some((local, loc))
    }

    /// 单次哈希桶探测提取槽位与物理位置并从索引移除（若后续防御校验失败可通过 `restore_index_entry` 无损回滚）
    #[inline]
    pub(crate) fn take_slot_and_location(
        &mut self,
        entity: EntityId,
    ) -> Option<(LocalSlotId, EntityLocation)> {
        if let Some((slot, meta)) = self.candidate_slot_meta(entity) {
            let loc = meta.location?;
            return Some((LocalSlotId::new(slot, meta.generation), loc));
        }
        if self.by_entity.is_empty() {
            return None;
        }
        let std::collections::hash_map::Entry::Occupied(occ) = self.by_entity.entry(entity) else {
            return None;
        };
        let local = *occ.get();
        let loc = self.slots.get(local.slot as usize)?.location?;
        occ.remove();
        Some((local, loc))
    }

    #[inline]
    pub(crate) fn restore_index_entry(&mut self, entity: EntityId, local: LocalSlotId) {
        if !Self::is_identity_slot(entity, local.slot) {
            self.by_entity.insert(entity, local);
        }
    }

    #[inline]
    pub(crate) fn finish_deallocate_slot_infallible(
        &mut self,
        local: LocalSlotId,
    ) -> Option<EntityLocation> {
        let meta = &mut self.slots[local.slot as usize];
        let location = meta.location.take();
        meta.entity = None;
        meta.generation += 1;
        self.free_slots.push(local.slot);
        location
    }

    #[inline]
    pub(crate) fn validate_deallocate_slot(&self, local: LocalSlotId) -> Result<(), StateError> {
        self.slots[local.slot as usize]
            .generation
            .checked_add(1)
            .ok_or(StateError::GenerationOverflow)?;
        Ok(())
    }

    #[inline]
    pub(crate) fn set_location_by_slot_infallible(
        &mut self,
        local: LocalSlotId,
        location: EntityLocation,
    ) {
        self.slots[local.slot as usize].location = Some(location);
    }

    #[inline]
    pub(crate) fn deallocate_slot_infallible(
        &mut self,
        entity: EntityId,
        local: LocalSlotId,
    ) -> Option<EntityLocation> {
        let meta = &mut self.slots[local.slot as usize];
        let location = meta.location.take();
        meta.entity = None;
        meta.generation += 1;
        if !Self::is_identity_slot(entity, local.slot) {
            self.by_entity.remove(&entity);
        }
        self.free_slots.push(local.slot);
        location
    }

    #[inline]
    pub fn local_slot(&self, entity: EntityId) -> Option<LocalSlotId> {
        if let Some((slot, meta)) = self.candidate_slot_meta(entity) {
            return Some(LocalSlotId::new(slot, meta.generation));
        }
        if self.by_entity.is_empty() {
            return None;
        }
        self.by_entity.get(&entity).copied()
    }

    pub fn entity_for_slot(&self, local: LocalSlotId) -> Option<EntityId> {
        let meta = self.slots.get(local.slot as usize)?;
        let entity = meta.entity?;
        if meta.generation != local.generation {
            return None;
        }
        if Self::is_identity_slot(entity, local.slot) || self.by_entity.get(&entity) == Some(&local)
        {
            Some(entity)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_alive(&self, entity: EntityId) -> bool {
        if self.candidate_slot_meta(entity).is_some() {
            return true;
        }
        !self.by_entity.is_empty() && self.by_entity.contains_key(&entity)
    }

    /// 针对已知代际槽位进行活跃性与代际校验，并提取当前物理位置（仅供包内受信操作与寻址差分对照使用）。
    ///
    /// 安全代际槽位寻址辅助函数（Layer B 路径），纯 O(1) 数组索引访问，跳过 UUID B 树：
    /// 1. `local.slot < slots.len()`；
    /// 2. `meta.generation == local.generation`（代际防 ABA 匹配）；
    /// 3. `meta.entity == Some(expected_entity)`（实体所有权匹配）；
    /// 4. `meta.location` 确实存在。
    #[inline]
    pub(crate) fn get_location_by_slot(
        &self,
        expected_entity: EntityId,
        local: LocalSlotId,
    ) -> Option<EntityLocation> {
        let meta = self.slots.get(local.slot as usize)?;
        if meta.generation != local.generation {
            return None;
        }
        if meta.entity != Some(expected_entity) {
            return None;
        }
        meta.location
    }

    #[inline]
    pub fn get_location(&self, entity: EntityId) -> Option<EntityLocation> {
        if let Some((_, meta)) = self.candidate_slot_meta(entity) {
            return meta.location;
        }
        if self.by_entity.is_empty() {
            return None;
        }
        let local = *self.by_entity.get(&entity)?;
        self.slots.get(local.slot as usize)?.location
    }

    #[inline]
    pub(crate) fn resolve_with_hint(
        &self,
        entity: EntityId,
        slot_hint: &mut usize,
    ) -> Option<(LocalSlotId, EntityLocation)> {
        let expected_slot = *slot_hint;
        if expected_slot < self.slots.len() {
            let meta = &self.slots[expected_slot];
            if meta.entity == Some(entity) {
                if let Some(loc) = meta.location {
                    let local = LocalSlotId::new(expected_slot as u32, meta.generation);
                    *slot_hint = expected_slot.saturating_add(1);
                    return Some((local, loc));
                }
            }
        }

        let local = self.local_slot(entity)?;
        let loc = self.slots[local.slot as usize].location?;
        let actual_slot = local.slot as usize;
        *slot_hint = actual_slot.saturating_add(1);
        Some((local, loc))
    }

    pub(crate) fn are_entities_consecutive_slots(&self, entities: &[EntityId]) -> bool {
        let Some(&first_id) = entities.first() else {
            return true;
        };
        let Some(first_local) = self.local_slot(first_id) else {
            return false;
        };
        entities.iter().enumerate().all(|(i, &id)| {
            self.local_slot(id).is_some_and(|local| {
                local.generation == first_local.generation
                    && local.slot as usize == first_local.slot as usize + i
            })
        })
    }

    pub(crate) fn align_archetype_slots(
        &mut self,
        arch_id: u32,
        old_entities: &[EntityId],
        new_entities: &[EntityId],
    ) {
        assert_eq!(old_entities.len(), new_entities.len());
        let mut target_slots: Vec<u32> = old_entities
            .iter()
            .map(|&id| {
                self.local_slot(id)
                    .expect("validated installed entity")
                    .slot
            })
            .collect();
        for &old_id in old_entities {
            self.by_entity.remove(&old_id);
        }
        target_slots.sort_unstable();
        for (row, (&entity_id, &slot_idx)) in
            new_entities.iter().zip(target_slots.iter()).enumerate()
        {
            let generation = self.slots[slot_idx as usize].generation;
            self.slots[slot_idx as usize].entity = Some(entity_id);
            self.slots[slot_idx as usize].location =
                Some(EntityLocation::new(arch_id, row as u32));
            if !Self::is_identity_slot(entity_id, slot_idx) {
                self.by_entity
                    .insert(entity_id, LocalSlotId::new(slot_idx, generation));
            }
        }
    }

    pub fn set_location(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
    ) -> Result<(), StateError> {
        let local = self
            .local_slot(entity)
            .ok_or_else(|| StateError::AccessDenied(format!("Entity {entity} is not installed")))?;
        self.slots[local.slot as usize].location = Some(location);
        Ok(())
    }

    pub(crate) fn set_location_infallible(&mut self, entity: EntityId, location: EntityLocation) {
        let local = self
            .local_slot(entity)
            .expect("validated entity must remain installed");
        self.slots[local.slot as usize].location = Some(location);
    }

    pub(crate) fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.slots.iter().filter_map(|m| m.entity)
    }

    pub(crate) fn validate(&self) -> Result<(), StateError> {
        if self.rebuild_index()? != self.by_entity {
            return Err(StateError::AccessDenied(
                "Entity identity/slot index is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn rebuild_index(&self) -> Result<EntityIndex<LocalSlotId>, StateError> {
        if self.slots.len() >= u32::MAX as usize {
            return Err(StateError::CapacityExhausted(
                "Entity slot capacity u32::MAX reached".into(),
            ));
        }
        let mut free = BTreeSet::new();
        for &slot in &self.free_slots {
            if slot as usize >= self.slots.len() || !free.insert(slot) {
                return Err(StateError::AccessDenied(format!(
                    "Invalid or duplicate free slot {slot}"
                )));
            }
        }
        let non_identity_estimate = self
            .slots
            .iter()
            .enumerate()
            .filter(|&(slot, meta)| {
                meta.entity
                    .is_some_and(|e| !Self::is_identity_slot(e, slot as u32))
            })
            .count();
        let mut index =
            EntityIndex::with_capacity_and_hasher(non_identity_estimate, Default::default());
        for (slot, meta) in self.slots.iter().enumerate() {
            if meta.generation == 0 {
                return Err(StateError::AccessDenied(format!(
                    "Slot {slot} has zero generation"
                )));
            }
            match meta.entity {
                Some(entity) => {
                    if entity.is_nil() || free.contains(&(slot as u32)) {
                        return Err(StateError::AccessDenied(format!(
                            "Invalid, duplicate or free-listed identity at slot {slot}"
                        )));
                    }
                    if Self::is_identity_slot(entity, slot as u32) {
                        if index.contains_key(&entity) {
                            return Err(StateError::AccessDenied(format!(
                                "Invalid, duplicate or free-listed identity at slot {slot}"
                            )));
                        }
                    } else if self.candidate_slot_meta(entity).is_some()
                        || index
                            .insert(entity, LocalSlotId::new(slot as u32, meta.generation))
                            .is_some()
                    {
                        return Err(StateError::AccessDenied(format!(
                            "Invalid, duplicate or free-listed identity at slot {slot}"
                        )));
                    }
                }
                None if !free.contains(&(slot as u32)) || meta.location.is_some() => {
                    return Err(StateError::AccessDenied(format!(
                        "Unoccupied slot {slot} is missing from the free list or retains a location"
                    )));
                }
                None => {}
            }
        }
        Ok(index)
    }

    pub fn active_count(&self) -> usize {
        self.slots.len().saturating_sub(self.free_slots.len())
    }
    pub fn len(&self) -> usize {
        self.active_count()
    }
    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }
    pub fn total_slots(&self) -> usize {
        self.slots.len()
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn index_capacity(&self) -> usize {
        self.by_entity.capacity()
    }
    /// 估算分配器核心容器在堆上占用的内存理论下界字节数。
    ///
    /// 包含：
    /// - `slots` 向量容量开销 (`capacity * size_of::<SlotMeta>()`)；
    /// - `free_slots` 向量容量开销；
    /// - `by_entity` 哈希索引桶与控制字节估算。
    ///
    /// 注意：
    /// 1. 此估算仅涵盖分配器自身，不包含组件列（Archetype columns）或全局 Store 元数据；
    /// 2. 标准库基于 hashbrown 的 HashMap，实际桶数向上取 2 的次幂且存在对齐与控制块开销，
    ///    故此处为结构性理论下界估算值，不可视作操作系统的精确常驻内存 (RSS)。
    pub fn heap_bytes_estimate(&self) -> usize {
        let slots_bytes = self.slots.capacity() * std::mem::size_of::<SlotMeta>();
        let free_bytes = self.free_slots.capacity() * std::mem::size_of::<u32>();
        let hashmap_bytes = self.by_entity.capacity()
            * (std::mem::size_of::<(EntityId, LocalSlotId)>() + 1);
        slots_bytes + free_bytes + hashmap_bytes
    }
}

#[cfg(test)]
#[path = "allocator_tests.rs"]
mod tests;
