use super::{NativeLanes, PlanScratch};
use crate::binding::EntityId;
use crate::id::ComponentTypeId;
use crate::storage::allocator::EntityLocation;
use crate::storage::archetype::Archetype;
use crate::storage::batch::planned_patch_values;
use crate::storage::batch::BatchReceipt;
use crate::storage::registry::ComponentDefinition;
use crate::storage::ErasedComponentValue;
use std::collections::BTreeMap;

/// 内部已校验的批次结构变更或创建/删除计划项
#[derive(Debug)]
pub(crate) enum PlannedOp {
    Create {
        entity: EntityId,
        target_archetype: u32,
        components: std::ops::Range<usize>,
    },
    CreatePooled {
        entity: EntityId,
        target_archetype: u32,
        pool_start: u32,
        perm: [u8; 8],
        len: u8,
    },
    StructuralPatch {
        entity: EntityId,
        location: EntityLocation,
        target_archetype: u32,
        new_components: std::ops::Range<usize>,
        replaced_components: std::ops::Range<usize>,
        removed_components: std::ops::Range<usize>,
        pool_index: u32,
    },
    Delete {
        entity: EntityId,
        location: EntityLocation,
        pool_index: u32,
    },
}

/// 批次执行计划的提交模式（收据保留 vs 丢弃）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedCommitMode {
    KeepReceipt,
    DiscardReceipt,
}

/// 已预校验的批次执行计划 (PreparedBatch)。
///
/// 具备不可复制（不实现 Clone/Copy）、防过期（绑定 epoch）与防跨实例（绑定 store_id）特性。
#[must_use = "prepared batch must be committed to apply changes"]
pub struct PreparedBatch {
    pub(crate) store_id: u64,
    pub(crate) epoch: u64,
    pub(crate) commit_mode: PreparedCommitMode,
    pub(crate) plan_scratch: PlanScratch,
    pub(crate) in_place_values: planned_patch_values::PlannedInPlaceValues,
    pub(crate) native_plans: NativeLanes,
    pub(crate) native_plan_count: usize,
    pub(crate) plans: Vec<PlannedOp>,
    pub(crate) structural_values: planned_patch_values::PlannedPatchValues,
    pub(crate) detached: Vec<Archetype>,
    pub(crate) new_definitions: BTreeMap<ComponentTypeId, ComponentDefinition>,
    pub(crate) receipt: BatchReceipt,
    // Destruction scratch for structural removals so all columns/locations finish
    // updating before user Drop code runs.
    pub(crate) discarded_values: Vec<ErasedComponentValue>,
}

impl std::fmt::Debug for PreparedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedBatch")
            .field("store_id", &self.store_id)
            .field("epoch", &self.epoch)
            .field("commit_mode", &self.commit_mode)
            .field("in_place_plan_count", &self.in_place_plan_count())
            .field("structural_plan_count", &self.plans.len())
            .finish_non_exhaustive()
    }
}

impl PreparedBatch {
    #[inline]
    pub fn store_id(&self) -> u64 {
        self.store_id
    }

    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn commit_mode(&self) -> PreparedCommitMode {
        self.commit_mode
    }

    #[inline]
    pub fn in_place_plan_count(&self) -> usize {
        self.plan_scratch.in_place_plans.len() + self.native_plan_count
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn in_place_plans_capacity(&self) -> usize {
        self.plan_scratch.in_place_plans.capacity()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn plans_capacity(&self) -> usize {
        self.plans.capacity()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn receipt_created_capacity(&self) -> usize {
        self.receipt.created_entities.capacity()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn receipt_updated_capacity(&self) -> usize {
        self.receipt.updated_entities.capacity()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn plans_len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn receipt_updated_len(&self) -> usize {
        self.receipt.updated_entities.len()
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn receipt_created_len(&self) -> usize {
        self.receipt.created_entities.len()
    }
}
