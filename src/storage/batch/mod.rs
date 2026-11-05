//! 存储批量事务与收据协议 (Batch & CommitReceipt)

mod encoded;
mod ops;
pub(crate) mod planned_patch_values;
mod store_batch;
mod store_exec;

/// 批次提交收据 (CommitReceipt)
///
/// 记录单次批量提交或命令缓冲提交产生的新纪元 (`epoch`) 以及新增、修改与删除的实体集合，
/// 供帧间增量查询 (`added_since_from_receipt` / `changed_since_from_receipt`) 使用。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommitReceipt {
    pub epoch: u64,
    pub created_entities: Vec<crate::id::EntityId>,
    pub updated_entities: Vec<crate::id::EntityId>,
    pub deleted_entities: Vec<crate::id::EntityId>,
}

impl CommitReceipt {
    #[inline]
    pub fn len(&self) -> usize {
        self.created_entities.len() + self.updated_entities.len() + self.deleted_entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type BatchReceipt = CommitReceipt;

pub use encoded::{
    EncodedColumnOpKind, EncodedColumnRun, EncodedStoreBatch, ENCODED_BATCH_FORMAT_V2,
};
pub use ops::{BatchOp, EntityBundle, EntityPatch, PatchOp};
pub use store_batch::StoreBatch;
pub use store_exec::PreparedBatch;

pub type EncodedColumnUpdate = EncodedColumnRun;
pub type ColumnUpdateBatch = StoreBatch;

#[allow(unused_imports)]
pub(crate) use ops::{
    CompactPatchOps, NativeUpdateRef, PatchSemantics, PoolEntry, TOMBSTONE_OP,
};
pub(crate) use crate::storage::core::column::InPlaceOp;
pub(crate) use store_batch::BatchOpCounts;
#[allow(unused_imports)]
pub(crate) use store_exec::{
    CommandEntries, CommandEntry, NativeDirectOps, NativeLane, NativeLanes, NativeType,
    PlanScratch, PlannedInPlacePatch, PlannedOp, PreparedCommitMode,
};

#[cfg(test)]
mod tests;
