//! 阶段 1：命令预检与批次准备门面 (Phase 1: Command Entry & Batch Preparation Facade)


use super::*;
use super::Store;
use crate::storage::archetype::Archetype;
use crate::id::ArchetypeId;
use crate::storage::batch::{
    BatchReceipt, CompactPatchOps, PatchOp,
    PreparedBatch, StoreBatch,
};
use crate::storage::core::column::{InPlaceReplacements, PlannedInPlacePatch};
use crate::storage::batch::planned_patch_values::{
    PlannedPatchValues, SmallList,
};
use crate::storage::column::AnyColumn;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use std::collections::BTreeMap;

use super::scratch::CommandEntry;
use super::planning::{
    unroll_planning_to_batch, PlanningInput, StructuralRouteKey, MAX_STRUCTURAL_ROUTES,
};
use super::validation::{
    ColumnCacheTracker, ColumnValidation,
};
use crate::storage::core::store::ArchetypeValidationBitmap;


impl Store {
    /// 阶段 1：校验命令条目并准备目标布局；只允许扩容，不发布逻辑内容、模式或版本变更。
    /// 直接从命令条目生成 PreparedBatch 计划，避免构造中间 StoreBatch 与二次实体索引。
    pub(crate) fn prepare_command_entries_with_receipt<const KEEP: bool>(
        &mut self,
        entries: impl PlanningInput,
    ) -> Result<PreparedBatch, StateError> {
        self.prepare_input_with_receipt::<KEEP>(entries)
    }

    pub(crate) fn prepare_input(&mut self, entries: impl PlanningInput) -> Result<PreparedBatch, StateError> {
        self.prepare_input_with_receipt::<true>(entries)
    }


    /// 阶段 1（可恢复输入）：单遍历原子校验并生成执行计划。若校验失败，全路径无损归还 StoreBatch。
    pub fn prepare_batch_recoverable(
        &mut self,
        batch: StoreBatch,
    ) -> Result<PreparedBatch, (StateError, StoreBatch)> {
        self.prepare_batch_with_receipt_recoverable::<true>(batch)
    }

    /// 阶段 1（丢弃收据且可恢复输入）：单遍历原子校验。若校验失败，全路径无损归还 StoreBatch。
    pub fn prepare_batch_discard_recoverable(
        &mut self,
        batch: StoreBatch,
    ) -> Result<PreparedBatch, (StateError, StoreBatch)> {
        self.prepare_batch_with_receipt_recoverable::<false>(batch)
    }

    fn prepare_batch_with_receipt_recoverable<const KEEP: bool>(
        &mut self,
        mut batch: StoreBatch,
    ) -> Result<PreparedBatch, (StateError, StoreBatch)> {
        let native_values = batch.take_native_values();
        let res = if native_values.is_empty() {
            self.prepare_input_values::<false, KEEP>(batch, native_values)
        } else {
            self.prepare_input_values::<true, KEEP>(batch, native_values)
        };
        res.map_err(|(e, recovered)| {
            (e, recovered.expect("StoreBatch is always recoverable"))
        })
    }

    /// 阶段 1：校验并准备目标布局；只允许扩容，不发布逻辑内容、模式或版本变更。
    pub fn prepare_batch(&mut self, batch: StoreBatch) -> Result<PreparedBatch, StateError> {
        self.prepare_batch_recoverable(batch).map_err(|(e, _)| e)
    }

    /// 阶段 1（丢弃收据）：校验并准备目标布局，预先跳过收据容器分配。
    #[allow(dead_code)]
    pub(crate) fn prepare_batch_discard(&mut self, batch: StoreBatch) -> Result<PreparedBatch, StateError> {
        self.prepare_batch_discard_recoverable(batch).map_err(|(e, _)| e)
    }
}

mod input;
mod preverify;


impl Store {
    pub(crate) fn prepare_input_with_receipt<const KEEP: bool>(
        &mut self,
        mut entries: impl PlanningInput,
    ) -> Result<PreparedBatch, StateError> {
        let native_values = entries.take_native_values();
        // Dispatch once per buffer, not once per entity. Both specializations
        // share the same checks/plans/receipt kernel; ordinary buffers need not
        // carry the native stream branch through their hot loop.
        let res = if native_values.is_empty() {
            self.prepare_input_values::<false, KEEP>(entries, native_values)
        } else {
            self.prepare_input_values::<true, KEEP>(entries, native_values)
        };
        res.map_err(|(e, _)| e)
    }
}

