use std::collections::{BTreeMap, BTreeSet};

use crate::binding::EntityId;
use crate::id::ComponentTypeId;
use crate::storage::batch::{
    BatchOp, CompactPatchOps, EntityBundle, EntityPatch, InPlaceOp, PatchOp,
    PatchSemantics, PlannedOp, PoolEntry, StoreBatch,
};
use crate::storage::core::column::{InPlaceReplacements, PlannedInPlacePatch};
use crate::storage::batch::planned_patch_values::{
    InPlaceValueSource, PlannedInPlaceValues, PlannedPatchValues,
};
use crate::storage::ErasedComponentValue;
use crate::storage::batch::BatchOpCounts;
use super::NativeLanes;
use super::scratch::{CommandEntries, CommandEntry};

#[cfg(test)]
thread_local! {
    pub(crate) static ROUTE_SIGNATURE_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A batch-local route uses the actual structural delta, not values or the
/// complete target signature. Common deltas stay on the stack. Keys are exact
/// (no hash-only matching) and include the source layout.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StructuralRouteKey {
    pub(crate) source: u32,
    pub(crate) changes: smallvec::SmallVec<[(ComponentTypeId, bool); 4]>,
}

impl StructuralRouteKey {
    pub(crate) fn new(source: u32) -> Self {
        Self { source, changes: smallvec::SmallVec::new() }
    }

    pub(crate) fn contains_after(&self, source: &BTreeSet<ComponentTypeId>, ty: ComponentTypeId) -> bool {
        self.changes.iter().find(|(id, _)| *id == ty)
            .map_or_else(|| source.contains(&ty), |(_, added)| *added)
    }

    pub(crate) fn target_types(&self, source: &BTreeSet<ComponentTypeId>) -> BTreeSet<ComponentTypeId> {
        #[cfg(test)]
        ROUTE_SIGNATURE_BUILDS.with(|count| count.set(count.get() + 1));
        let mut target = source.clone();
        for &(ty, added) in &self.changes {
            if added { target.insert(ty); } else { target.remove(&ty); }
        }
        target
    }

    // Bound retained scratch independently of entity count and avoid retaining
    // heap-backed, unusually large deltas. Uncached routes use normal planning.
    pub(crate) fn cacheable(&self) -> bool { !self.changes.spilled() }
}

pub(crate) const MAX_STRUCTURAL_ROUTES: usize = 64;

/// Custom zero-allocation iterator for StoreBatch entries
pub(crate) struct StoreBatchIntoEntries {
    pub(crate) iter: std::collections::btree_map::IntoIter<EntityId, BatchOp>,
}

impl Iterator for StoreBatchIntoEntries {
    type Item = (EntityId, CommandEntry);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (id, op) = self.iter.next()?;
        let entry = match op {
            BatchOp::CreateEntity { bundle, .. } => CommandEntry::Create(bundle),
            BatchOp::PatchEntity { patch, .. } => CommandEntry::Patch(patch),
            BatchOp::DeleteEntity { .. } => CommandEntry::Delete,
        };
        Some((id, entry))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for StoreBatchIntoEntries {}

/// Private, owning input adapters. Preflight borrows the original containers;
/// planning then consumes them without rebuilding an entity index.
pub(crate) trait PlanningInput: Sized {
    type Iter<'a>: Iterator<Item = (EntityId, CommandEntry)> where Self: 'a;
    fn take_native_values(&mut self) -> NativeLanes {
        Default::default()
    }
    fn len(&self) -> usize;
    fn pool_len(&self) -> usize {
        0
    }
    fn counts(&self) -> BatchOpCounts;
    fn has_only_native_updates(&self) -> bool { false }
    fn for_each_value(&self, f: impl FnMut(&ErasedComponentValue));
    // Drain values while retaining the owning containers for bounded reuse.
    fn drain_entries(&mut self) -> (Self::Iter<'_>, &mut [PoolEntry]);
    fn mark_pool_consumed(&mut self) {}
    fn take_pool(&mut self) -> Vec<PoolEntry> {
        Vec::new()
    }
    fn has_only_wide_patches(&self) -> bool;
    fn is_recoverable_batch(&self) -> bool {
        false
    }
}

impl PlanningInput for StoreBatch {
    type Iter<'a> = StoreBatchIntoEntries;

    fn len(&self) -> usize {
        self.len()
    }

    fn counts(&self) -> BatchOpCounts {
        self.counts()
    }

    fn for_each_value(&self, mut f: impl FnMut(&ErasedComponentValue)) {
        for op in self.ops().values() {
            op.for_each_value(&mut f);
        }
    }

    #[inline]
    fn drain_entries(&mut self) -> (Self::Iter<'_>, &mut [PoolEntry]) {
        (StoreBatchIntoEntries {
            iter: std::mem::take(self).into_ops().into_iter(),
        }, &mut [])
    }

    fn has_only_wide_patches(&self) -> bool {
        self.ops().values().all(|op| matches!(op,
            BatchOp::PatchEntity { patch, .. } if patch.is_empty() || patch.ops().len() >= 3))
    }

    fn is_recoverable_batch(&self) -> bool {
        true
    }
}

impl PlanningInput for CommandEntries {
    type Iter<'a> = std::vec::Drain<'a, (EntityId, CommandEntry)>;

    fn take_native_values(&mut self) -> NativeLanes {
        std::mem::take(&mut self.native_values)
    }

    #[inline]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    fn pool_len(&self) -> usize {
        self.pool.len()
    }

    #[inline]
    fn counts(&self) -> BatchOpCounts {
        self.counts
    }

    fn has_only_native_updates(&self) -> bool {
        self.entries.iter().all(|(_, entry)| matches!(entry, CommandEntry::NativeUpdate(_)))
    }

    #[inline]
    fn for_each_value(&self, mut f: impl FnMut(&ErasedComponentValue)) {
        for (_, entry) in &self.entries {
            match entry {
                CommandEntry::Create(bundle) => {
                    for v in bundle.components().values() {
                        f(v);
                    }
                    bundle.for_each_pooled_value(&self.pool, &mut f);
                }
                CommandEntry::Patch(patch) => {
                    if let CompactPatchOps::Pooled { head, len, .. } = patch.ops() {
                        if *len > 0 {
                            let mut curr = *head;
                            while curr != u32::MAX {
                                let pe = &self.pool[curr as usize];
                                match &pe.op {
                                    PatchOp::Insert(v)
                                    | PatchOp::Replace(v)
                                    | PatchOp::Upsert(v) => f(v),
                                    PatchOp::Remove(_) => {}
                                }
                                curr = pe.next;
                            }
                        }
                    } else {
                        entry.for_each_value(&mut f);
                    }
                }
                _ => entry.for_each_value(&mut f),
            }
        }
    }

    #[inline]
    fn drain_entries(&mut self) -> (Self::Iter<'_>, &mut [PoolEntry]) {
        (self.entries.drain(..), &mut self.pool)
    }

    #[inline]
    fn mark_pool_consumed(&mut self) {
        unsafe {
            self.pool.set_len(0);
        }
    }

    #[inline]
    fn take_pool(&mut self) -> Vec<PoolEntry> {
        std::mem::take(&mut self.pool)
    }

    fn has_only_wide_patches(&self) -> bool {
        self.entries.iter().all(|(_, entry)| matches!(entry,
            CommandEntry::Patch(patch) if patch.is_empty() || patch.ops().len() >= 3))
    }
}

/// 准备阶段失败时的执行计划无损反向重构引擎。
/// 零拷贝回收已分配的计划组件与未消费的条目，重构成完整的 StoreBatch 归还调用者。
pub(crate) fn unroll_planning_to_batch(
    in_place_plans: &mut Vec<PlannedInPlacePatch>,
    in_place_values: PlannedInPlaceValues,
    plans: Vec<PlannedOp>,
    mut structural_values: PlannedPatchValues,
    current_entry: Option<(EntityId, CommandEntry)>,
    remaining_entries: impl Iterator<Item = (EntityId, CommandEntry)>,
) -> StoreBatch {
    let mut ops = BTreeMap::new();
    let mut in_place_values = in_place_values.into_values();

    for plan in in_place_plans.drain(..) {
        let mut patch = EntityPatch::new();
        match plan.replacements {
            InPlaceReplacements::Single(_cid, op) => match op {
                InPlaceOp::Replace(val) => {
                    let _ = patch.replace_erased(val);
                }
                InPlaceOp::Upsert(val) => {
                    let _ = patch.upsert_erased(val);
                }
            },
            InPlaceReplacements::Multiple(ops_vec) => {
                for (_cid, op) in in_place_values.take_range(ops_vec) {
                    match op {
                        InPlaceOp::Replace(val) => {
                            let _ = patch.replace_erased(val);
                        }
                        InPlaceOp::Upsert(val) => {
                            let _ = patch.upsert_erased(val);
                        }
                    }
                }
            }
            InPlaceReplacements::Empty => {}
            InPlaceReplacements::Takeover(_, _) => {
                unreachable!("takeover replacements are not used in contiguous mode")
            }
            InPlaceReplacements::NativeMultiple(_) => {
                unreachable!("recoverable StoreBatch contains no typed lanes")
            }
        }
        ops.insert(plan.entity, BatchOp::PatchEntity { id: plan.entity, patch });
    }

    for plan in plans {
        match plan {
            PlannedOp::Create { entity, components, .. } => {
                let map: BTreeMap<ComponentTypeId, ErasedComponentValue> =
                    structural_values.drain_range(components).into_iter().collect();
                let bundle = EntityBundle::from_components(map);
                ops.insert(entity, BatchOp::CreateEntity { id: entity, bundle });
            }
            PlannedOp::CreatePooled { .. } => {
                unreachable!("recoverable StoreBatch contains no pooled creates")
            }
            PlannedOp::StructuralPatch {
                entity,
                new_components,
                replaced_components,
                removed_components,
                ..
            } => {
                let mut patch = EntityPatch::new();
                for (_cid, val) in structural_values.drain_range(new_components) {
                    let _ = patch.insert_erased(val);
                }
                for (_cid, val) in structural_values.drain_range(replaced_components) {
                    let _ = patch.replace_erased(val);
                }
                for &cid in structural_values.removed_range(removed_components) {
                    let _ = patch.remove_by_id(cid, "");
                }
                ops.insert(entity, BatchOp::PatchEntity { id: entity, patch });
            }
            PlannedOp::Delete { entity, .. } => {
                ops.insert(entity, BatchOp::DeleteEntity { id: entity });
            }
        }
    }

    if let Some((id, entry)) = current_entry {
        match entry {
            CommandEntry::Create(bundle) => {
                ops.insert(id, BatchOp::CreateEntity { id, bundle });
            }
            CommandEntry::Patch(patch) => {
                ops.insert(id, BatchOp::PatchEntity { id, patch });
            }
            CommandEntry::Delete => {
                ops.insert(id, BatchOp::DeleteEntity { id });
            }
            CommandEntry::NativeUpdate(_) => {}
        }
    }

    for (id, entry) in remaining_entries {
        match entry {
            CommandEntry::Create(bundle) => {
                ops.insert(id, BatchOp::CreateEntity { id, bundle });
            }
            CommandEntry::Patch(patch) => {
                ops.insert(id, BatchOp::PatchEntity { id, patch });
            }
            CommandEntry::Delete => {
                ops.insert(id, BatchOp::DeleteEntity { id });
            }
            CommandEntry::NativeUpdate(_) => {}
        }
    }

    StoreBatch::from_ops(ops)
}

impl PlanningInput for (EntityId, CommandEntry) {
    type Iter<'a> = std::iter::Once<(EntityId, CommandEntry)>;

    fn len(&self) -> usize {
        1
    }

    fn counts(&self) -> BatchOpCounts {
        let mut counts = BatchOpCounts::default();
        match &self.1 {
            CommandEntry::Create(_) => counts.creates = 1,
            CommandEntry::Delete => counts.deletes = 1,
            CommandEntry::NativeUpdate(_) => unreachable!("native lanes require CommandEntries"),
            CommandEntry::Patch(patch) => match patch.semantics() {
                PatchSemantics::Empty => counts.empty_patches = 1,
                PatchSemantics::InPlace => counts.in_place_patches = 1,
                PatchSemantics::Structural => counts.structural_patches = 1,
                PatchSemantics::Conditional => counts.conditional_patches = 1,
            },
        }
        counts
    }

    fn for_each_value(&self, f: impl FnMut(&ErasedComponentValue)) {
        self.1.for_each_value(f);
    }

    #[inline]
    fn drain_entries(&mut self) -> (Self::Iter<'_>, &mut [PoolEntry]) {
        let entry = std::mem::replace(&mut self.1, CommandEntry::Delete);
        (std::iter::once((self.0, entry)), &mut [])
    }

    fn has_only_wide_patches(&self) -> bool {
        matches!(&self.1, CommandEntry::Patch(patch) if patch.ops().len() >= 3)
    }
}
