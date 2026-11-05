//! 延迟命令缓冲区与流式实体修改器 (CommandBuffer & EntityCommands)
//!
//! 允许在只读遍历实体或复杂逻辑分支中缓冲增删改意图，
//! 并在迭代结束后统一提交两阶段事务批次，保证代际安全与严格冲突校验。
//!
//! ```compile_fail
//! // CommandBuffer::new is crate-private
//! use sumeru::CommandBuffer;
//! let _ = CommandBuffer::new(1, 1);
//! ```
//!
//! ```compile_fail
//! // CommandBuffer::into_batch is crate-private
//! use sumeru::Store;
//! let store = Store::new();
//! let cmd = store.commands();
//! let _ = cmd.into_batch();
//! ```
//!
//! Homogeneous recording is a CommandBuffer operation, not another public batch.
//! ```compile_fail
//! use sumeru::UpdateBatch;
//! ```
//! ```compile_fail
//! let store = sumeru::Store::new();
//! let _ = store.update_batch::<u64>();
//! ```
//! ```compile_fail
//! let mut store = sumeru::Store::new();
//! let _ = store.update_many::<u64>([]);
//! ```
//! ```compile_fail
//! let mut store = sumeru::Store::new();
//! let _ = store.modify_batch::<u64>([], |_, _| {});
//! ```
//!
use crate::storage::batch::{BatchReceipt, EntityBundle, EntityPatch, PatchSemantics, StoreBatch};
use crate::storage::ErasedComponentValue;
use crate::storage::entity::IntoEntity;
use crate::storage::store::Store;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::hash_map::Entry;
use crate::storage::core::entity_index::EntityIndex;

pub(crate) mod updates;

pub use updates::BatchUpdateItem;

use crate::storage::batch::{
    BatchOpCounts, CommandEntries, CommandEntry, NativeDirectOps, NativeLane,
    NativeLanes, NativeType, NativeUpdateRef,
};

#[inline]
pub(crate) fn sync_index(
    entries: &[(EntityId, CommandEntry)],
    index: &mut EntityIndex<usize>,
) {
    if index.len() < entries.len() {
        index.reserve(entries.len() - index.len());
        for (i, (eid, _)) in entries.iter().enumerate().skip(index.len()) {
            index.insert(*eid, i);
        }
    }
}

impl Store {
    /// 创建绑定当前存储库实例与代际的延迟命令缓冲区 (`CommandBuffer`)。
    #[inline]
    pub fn commands(&self) -> CommandBuffer {
        CommandBuffer::new_with_limits(
            self.store_id(),
            self.epoch(),
            CommandLimits::default(),
        )
    }

    /// 创建带有指定限额保护的延迟命令缓冲区 (`CommandBuffer`)。
    #[inline]
    pub fn commands_with_limits(&self, limits: CommandLimits) -> CommandBuffer {
        CommandBuffer::new_with_limits(
            self.store_id(),
            self.epoch(),
            limits,
        )
    }
}

/// 命令缓冲限额配置。
///
/// 用于限制单次命令缓冲区允许记录的实体总数及组件级操作总数，防止失控批次耗尽内存。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommandLimits {
    /// 允许记录的唯一实体总数上限（`None` 表示不限，`Some(0)` 表示不允许记录任何实体）
    pub max_entities: Option<usize>,
    /// 允许记录的组件操作总数上限（创建 bundle 中组件数 + patch 操作数，`None` 表示不限，`Some(0)` 表示不允许记录组件操作）
    pub max_component_ops: Option<usize>,
}

/// 事务性延迟命令缓冲区 (CommandBuffer)。
///
/// 允许通过流畅 API（Fluent API）记录跨实体的增删改命令，具备以下特性：
/// 1. **零中间批次与两阶段直通**：直接将命令缓冲移交至底层 `prepare_input` 生成 `PreparedBatch`，消除临时集合与二次校验；
/// 2. **首错熔断与容量防护**：记录阶段一旦违反限额配置（`CommandLimits`）或产生语法/冲突错误，立即冻结后续操作并在此后的 `prepare/apply` 中显式抛出；
/// 3. **同实体变更折叠**：多次调用同一实体的 `entity(id)` 将自动合并至同一补丁项，且对重复修改同一组件的操作施加严格检查；
/// 4. **轻量收据视图**：记录批次产生的 `epoch` 以及创建、修改与删除的实体列表。
#[derive(Debug)]
#[must_use = "CommandBuffer must be applied or prepared; dropping it discards staged mutations"]
pub struct CommandBuffer {
    store_id: u64,
    epoch: u64,
    limits: CommandLimits,
    pub(crate) component_op_count: usize,
    creates: usize,
    deletes: usize,
    in_place_patches: usize,
    structural_patches: usize,
    conditional_patches: usize,
    empty_patches: usize,
    pub(crate) entries: Vec<(EntityId, CommandEntry)>,
    pub(crate) index: EntityIndex<usize>,
    pub(crate) pool: Vec<crate::storage::batch::PoolEntry>,
    native_values: NativeLanes,
    deferred_error: Option<StateError>,
}

impl CommandBuffer {
    /// 创建绑定指定 store_id 与 epoch 的命令缓冲（包内受控构造）
    #[allow(dead_code)]
    pub(crate) fn new(store_id: u64, epoch: u64) -> Self {
        Self::new_with_limits(store_id, epoch, CommandLimits::default())
    }

    /// 创建带有限额配置的命令缓冲（包内受控构造）
    pub(crate) fn new_with_limits(store_id: u64, epoch: u64, limits: CommandLimits) -> Self {
        Self {
            store_id,
            epoch,
            limits,
            component_op_count: 0,
            creates: 0,
            deletes: 0,
            in_place_patches: 0,
            structural_patches: 0,
            conditional_patches: 0,
            empty_patches: 0,
            entries: Vec::new(),
            index: EntityIndex::default(),
            pool: Vec::new(),
            native_values: Default::default(),
            deferred_error: None,
        }
    }

    /// 限额配置
    #[inline]
    pub fn limits(&self) -> CommandLimits {
        self.limits
    }

    /// 绑定的存储实例 ID
    #[inline]
    pub fn store_id(&self) -> u64 {
        self.store_id
    }

    /// 绑定的存储代际 (Epoch)
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// 命令缓冲是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 涉及变更的实体总数
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 遍历命令缓冲中的所有写入值
    pub fn for_each_value(&self, mut f: impl FnMut(&crate::storage::ErasedComponentValue)) {
        for (_, entry) in &self.entries {
            match entry {
                CommandEntry::Create(bundle) => {
                    for v in bundle.components().values() {
                        f(v);
                    }
                    bundle.for_each_pooled_value(&self.pool, &mut f);
                }
                CommandEntry::Patch(patch) => {
                    if let &crate::storage::batch::CompactPatchOps::Pooled { head, len, .. } = patch.ops() {
                        if len > 0 {
                            let mut curr = head;
                            while curr != u32::MAX {
                                let pe = &self.pool[curr as usize];
                                match &pe.op {
                                    crate::storage::batch::PatchOp::Insert(v)
                                    | crate::storage::batch::PatchOp::Replace(v)
                                    | crate::storage::batch::PatchOp::Upsert(v) => f(v),
                                    crate::storage::batch::PatchOp::Remove(_) => {}
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

    /// 预留容量以减少录入期间的重分配。
    ///
    /// # 安全与防护保证
    /// 1. **首错短路**：若命令缓冲此前已处于错误状态，立即短路，不执行任何扩容；
    /// 2. **算术溢出防护**：检查 `self.entries.len().checked_add(additional)`，溢出时记录 `CapacityExhausted` 并拒绝扩容；
    /// 3. **限额配额保护**：若配置了 `limits.max_entities`，目标容量若超过配额上限，记录 `CapacityExhausted` 并拒绝扩容；
    /// 4. 仅在校验合法且未超限时，对底层条目向量与辅助哈希索引安全执行 `reserve(additional)`。
    pub fn reserve(&mut self, additional: usize) -> &mut Self {
        if self.deferred_error.is_some() {
            return self;
        }
        let target_len = match self.entries.len().checked_add(additional) {
            Some(len) => len,
            None => {
                self.record_error(StateError::CapacityExhausted(
                    "Command buffer reserve entity count overflowed".into(),
                ));
                return self;
            }
        };
        if u32::try_from(target_len).is_err() {
            self.record_error(StateError::CapacityExhausted("Command buffer entity capacity exceeds storage index range".into()));
            return self;
        }
        if let Some(max_entities) = self.limits.max_entities {
            if target_len > max_entities {
                self.record_error(StateError::CapacityExhausted(format!(
                    "Command buffer reserve of {additional} entities (target {target_len}) exceeds entity limit {max_entities}"
                )));
                return self;
            }
        }
        self.entries.reserve(additional);
        self.pool.reserve(additional.saturating_mul(4));
        if !self.index.is_empty() {
            self.index.reserve(additional);
        }
        self
    }

    pub(crate) fn record_error(&mut self, err: StateError) {
        if self.deferred_error.is_none() {
            self.deferred_error = Some(err);
        }
    }

    /// 创建实体并赋予初始组件束（支持 `Entity` 或 `(EntityId, (T1, ..., Tn))` 多元组）。
    pub fn insert(&mut self, input: impl IntoEntity) -> &mut Self {
        if self.deferred_error.is_some() {
            return self;
        }
        match input.into_entity() {
            Ok(entity) => {
                let (id, bundle) = entity.into_parts();
                self.create_entity(id, bundle)
            }
            Err(err) => {
                self.record_error(err);
                self
            }
        }
    }

    /// 批量录制实体整包创建（支持 `Entity` 或 `(EntityId, (T1, ..., Tn))` 多元组序列）。
    pub fn insert_batch<I, B>(&mut self, batch: I) -> &mut Self
    where
        I: IntoIterator<Item = B>,
        B: IntoEntity,
    {
        if self.deferred_error.is_some() {
            return self;
        }
        let iter = batch.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.reserve(lower);
        }
        for item in iter {
            if self.deferred_error.is_some() {
                break;
            }
            self.insert(item);
        }
        self
    }

    /// 排队创建一个空实体（后续可通过 `cmd.entity(id).set(component)` 追加组件）
    #[inline]
    pub fn create_empty(&mut self, id: EntityId) -> &mut Self {
        self.create_entity(id, EntityBundle::new())
    }

    /// 创建实体并赋予初始组件束
    pub fn create_entity(&mut self, id: EntityId, bundle: EntityBundle) -> &mut Self {
        if self.deferred_error.is_some() {
            return self;
        }
        let comp_count = bundle.components().len();
        let new_ops = match self.component_op_count.checked_add(comp_count) {
            Some(ops) => ops,
            None => {
                self.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        };
        if let Some(max_ops) = self.limits.max_component_ops {
            if new_ops > max_ops {
                self.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        }
        let Self {
            limits,
            entries,
            index,
            deferred_error,
            component_op_count,
            creates,
            ..
        } = self;
        if index.is_empty() && entries.last().is_none_or(|(last_id, _)| *last_id < id) {
            if let Some(max_entities) = limits.max_entities {
                if entries.len() >= max_entities {
                    if deferred_error.is_none() {
                        *deferred_error = Some(StateError::CapacityExhausted(
                            "Command buffer entity limit exceeded".into(),
                        ));
                    }
                    return self;
                }
            }
            *component_op_count = new_ops;
            *creates += 1;
            entries.push((id, CommandEntry::Create(bundle)));
            return self;
        }
        sync_index(entries, index);
        match index.entry(id) {
            Entry::Occupied(occ) => {
                let idx = *occ.get();
                match &entries[idx].1 {
                    CommandEntry::Delete => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} is already marked for deletion in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Patch(_) | CommandEntry::NativeUpdate(_) => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} is already marked for patching in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Create(_) => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} is already marked for creation in this command buffer"
                            )));
                        }
                    }
                }
            }
            Entry::Vacant(vac) => {
                if let Some(max_entities) = limits.max_entities {
                    if entries.len() >= max_entities {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::CapacityExhausted(
                                "Command buffer entity limit exceeded".into(),
                            ));
                        }
                        return self;
                    }
                }
                *component_op_count = new_ops;
                *creates += 1;
                let idx = entries.len();
                entries.push((id, CommandEntry::Create(bundle)));
                vac.insert(idx);
            }
        }
        self
    }

    /// 标记删除实体（`despawn` 的兼容别名）
    pub fn delete_entity(&mut self, id: EntityId) -> &mut Self {
        if self.deferred_error.is_some() {
            return self;
        }
        let Self {
            limits,
            entries,
            index,
            deferred_error,
            deletes,
            ..
        } = self;
        if index.is_empty() && entries.last().is_none_or(|(last_id, _)| *last_id < id) {
            if let Some(max_entities) = limits.max_entities {
                if entries.len() >= max_entities {
                    if deferred_error.is_none() {
                        *deferred_error = Some(StateError::CapacityExhausted(
                            "Command buffer entity limit exceeded".into(),
                        ));
                    }
                    return self;
                }
            }
            *deletes += 1;
            entries.push((id, CommandEntry::Delete));
            return self;
        }
        sync_index(entries, index);
        match index.entry(id) {
            Entry::Occupied(occ) => {
                let idx = *occ.get();
                match &entries[idx].1 {
                    CommandEntry::Delete => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} is already marked for deletion in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Patch(_) | CommandEntry::NativeUpdate(_) => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} cannot be deleted after being patched in the same buffer"
                            )));
                        }
                    }
                    CommandEntry::Create(_) => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Entity {id} cannot be deleted after being created in the same buffer"
                            )));
                        }
                    }
                }
            }
            Entry::Vacant(vac) => {
                if let Some(max_entities) = limits.max_entities {
                    if entries.len() >= max_entities {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::CapacityExhausted(
                                "Command buffer entity limit exceeded".into(),
                            ));
                        }
                        return self;
                    }
                }
                *deletes += 1;
                let idx = entries.len();
                entries.push((id, CommandEntry::Delete));
                vac.insert(idx);
            }
        }
        self
    }

    /// 标记销毁实体（`delete_entity` 的语义化统一主名）
    #[inline]
    pub fn despawn(&mut self, id: EntityId) -> &mut Self {
        self.delete_entity(id)
    }

    /// 获取针对特定实体的流式命令构建器
    #[inline]
    pub fn entity(&mut self, id: EntityId) -> EntityCommands<'_> {
        EntityCommands { buffer: self, id, entry_index: None }
    }

    /// 转换为底层批次，若存在语法或冲突错误将在此暴露（包内受控转换）
    #[allow(dead_code)]
    pub(crate) fn into_batch(mut self) -> Result<StoreBatch, StateError> {
        if let Some(err) = self.deferred_error.take() {
            return Err(err);
        }
        self.lower_native_updates();
        let mut batch = StoreBatch::new();
        for (id, entry) in self.entries.drain(..) {
            match entry {
                CommandEntry::Create(mut bundle) => {
                    bundle.detach_pool(&mut self.pool);
                    batch.create_entity(id, bundle)?;
                }
                CommandEntry::Patch(mut patch) => {
                    patch.detach_pool(&mut self.pool);
                    batch.patch_entity(id, patch)?;
                }
                CommandEntry::Delete => {
                    batch.delete_entity(id)?;
                }
                CommandEntry::NativeUpdate(_) => unreachable!("lowered recording"),
            }
        }
        Ok(batch)
    }

    /// 执行五步严格前置检查并生成 PreparedBatch 计划（不提交变更）：
    /// 1. 实例归属校验：`store.store_id() == self.store_id`；
    /// 2. 记录时代际校验：`store.epoch() == self.epoch`；
    /// 3. 缓冲首错暴露：返回记录阶段产生的语法/冲突/限额错误；
    /// 4. 直接传入内部规划核心生成执行计划（零中间批次与二次索引构建）；
    /// 5. 完成目标原型布局准备与容量预留。
    #[inline]
    pub fn prepare(self, store: &mut Store) -> Result<crate::storage::batch::PreparedBatch, StateError> {
        self.prepare_with_receipt::<true>(store)
    }

    /// 仅限内部与测试使用：执行五步严格前置检查并生成不保留收据的 PreparedBatch 计划（不提交变更）。
    ///
    /// 严禁传入保留收据的 `commit_batch`，必须与 `commit_batch_discard` 匹配。
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn prepare_discard(self, store: &mut Store) -> Result<crate::storage::batch::PreparedBatch, StateError> {
        self.prepare_with_receipt::<false>(store)
    }

    fn prepare_with_receipt<const KEEP: bool>(
        mut self,
        store: &mut Store,
    ) -> Result<crate::storage::batch::PreparedBatch, StateError> {
        store.check_poisoned()?;
        if store.store_id() != self.store_id {
            return Err(StateError::StoreMismatch {
                expected_store: self.store_id,
                actual_store: store.store_id(),
            });
        }
        if store.epoch() != self.epoch {
            return Err(StateError::StaleBatch {
                expected_epoch: self.epoch,
                actual_epoch: store.epoch(),
            });
        }
        if let Some(err) = self.deferred_error.take() {
            return Err(err);
        }
        let entries = std::mem::take(&mut self.entries);
        let pool = std::mem::take(&mut self.pool);
        let native_values = std::mem::take(&mut self.native_values);
        let prepared = store.prepare_command_entries_with_receipt::<KEEP>(CommandEntries {
            entries,
            index: std::mem::take(&mut self.index),
            pool,
            native_values,
            counts: BatchOpCounts {
                creates: self.creates,
                deletes: self.deletes,
                in_place_patches: self.in_place_patches,
                structural_patches: self.structural_patches,
                conditional_patches: self.conditional_patches,
                empty_patches: self.empty_patches,
            },
        })?;
        Ok(prepared)
    }

    /// 获取命令缓冲中各类操作统计（供容量规划与测试断言白盒验证）
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn counts(&self) -> BatchOpCounts {
        BatchOpCounts {
            creates: self.creates,
            deletes: self.deletes,
            in_place_patches: self.in_place_patches,
            structural_patches: self.structural_patches,
            conditional_patches: self.conditional_patches,
            empty_patches: self.empty_patches,
        }
    }

    /// 提交缓冲中的所有命令至目标存储，执行严格两阶段验证与代际提交。
    #[inline]
    pub fn apply(self, store: &mut Store) -> Result<BatchReceipt, StateError> {
        let prepared = self.prepare(store)?;
        store.commit_batch(prepared)
    }

    /// Apply with the same preflight, conflict, quota and version checks as `apply`,
    /// but without an undo receipt. Existing components are assigned directly;
    /// their previous owned resources are dropped normally, never cloned.
    ///
    /// Returned validation errors leave logical state unchanged. A component's
    /// panicking destructor is not a rollback boundary: writes may be partial,
    /// but the epoch is invalidated before writing. Removed structural values
    /// are dropped only after rows and entity locations are consistent.
    // Keep this batch-level orchestration in the library: caller-dependent
    // inlining under size-optimized LTO can distort the discard hot path.
    #[inline(never)]
    pub fn apply_discard(self, store: &mut Store) -> Result<(), StateError> {
        let prepared = self.prepare_with_receipt::<false>(store)?;
        store.commit_batch_discard(prepared)
    }
}

/// 针对单一实体的流式命令操作包装器 (EntityCommands)。
pub struct EntityCommands<'a> {
    buffer: &'a mut CommandBuffer,
    id: EntityId,
    // Command entry, not a physical entity location. The exclusive buffer borrow
    // prevents other recorders from changing the entries during this handle's life.
    entry_index: Option<usize>,
}

impl<'a> EntityCommands<'a> {
    /// 获取当前操作的实体 ID
    #[inline]
    pub fn id(&self) -> EntityId {
        self.id
    }

    #[inline]
    fn mutate<T: ModelComponent, F>(&mut self, op: F) -> &mut Self
    where
        F: FnOnce(&mut EntityPatch, &mut Vec<crate::storage::batch::PoolEntry>) -> Result<(), StateError>,
    {
        if self.buffer.deferred_error.is_some() {
            return self;
        }
        let new_ops = match self.buffer.component_op_count.checked_add(1) {
            Some(ops) => ops,
            None => {
                self.buffer.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        };
        if let Some(max_ops) = self.buffer.limits.max_component_ops {
            if new_ops > max_ops {
                self.buffer.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        }
        let id = self.id;
        let CommandBuffer {
            limits,
            entries,
            index,
            pool,
            deferred_error,
            component_op_count,
            in_place_patches,
            structural_patches,
            conditional_patches,
            empty_patches,
            native_values,
            ..
        } = &mut *self.buffer;
        let slot = match self.entry_index {
            Some(idx) => Ok(idx),
            None if entries.last().is_some_and(|(last_id, _)| *last_id == id) => {
                Ok(entries.len() - 1)
            }
            None if index.is_empty() && entries.last().is_none_or(|(last_id, _)| *last_id < id) => {
                Err(None)
            }
            None => {
                sync_index(entries, index);
                match index.entry(id) {
                    Entry::Occupied(occ) => Ok(*occ.get()),
                    Entry::Vacant(vac) => Err(Some(vac)),
                }
            }
        };
        match slot {
            Ok(idx) => {
                self.entry_index = Some(idx);
                let native_head = entries[idx].1.native_head();
                if native_head.is_some()
                    && updates::contains_type(native_head, native_values, crate::ComponentTypeId::of::<T>())
                {
                    *deferred_error = Some(StateError::ComponentDuplicateInPatch {
                        entity: id,
                        type_name: std::any::type_name::<T>().into(),
                    });
                    return self;
                }
                if let CommandEntry::NativeUpdate(head) = entries[idx].1 {
                    let mut patch = EntityPatch::new();
                    patch.native_head = Some(head);
                    entries[idx].1 = CommandEntry::Patch(patch);
                }
                match &mut entries[idx].1 {
                    CommandEntry::Delete => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Cannot modify entity {id} marked for deletion in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Create(_) => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Cannot patch entity {id} marked for creation in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Patch(patch) => {
                        let old_sem = if native_head.is_some() && patch.is_empty() {
                            PatchSemantics::InPlace
                        } else {
                            patch.semantics()
                        };
                        if let Err(e) = op(patch, pool) {
                            let mapped_err = match e {
                                StateError::ComponentDuplicateInPatch { type_name, .. } => {
                                    StateError::ComponentDuplicateInPatch {
                                        entity: id,
                                        type_name,
                                    }
                                }
                                other => other,
                            };
                            if deferred_error.is_none() {
                                *deferred_error = Some(mapped_err);
                            }
                        } else {
                            *component_op_count = new_ops;
                            let new_sem = patch.semantics();
                            if old_sem != new_sem {
                                match old_sem {
                                    PatchSemantics::Empty => *empty_patches -= 1,
                                    PatchSemantics::InPlace => *in_place_patches -= 1,
                                    PatchSemantics::Structural => *structural_patches -= 1,
                                    PatchSemantics::Conditional => *conditional_patches -= 1,
                                }
                                match new_sem {
                                    PatchSemantics::Empty => *empty_patches += 1,
                                    PatchSemantics::InPlace => *in_place_patches += 1,
                                    PatchSemantics::Structural => *structural_patches += 1,
                                    PatchSemantics::Conditional => *conditional_patches += 1,
                                }
                            }
                        }
                    }
                    CommandEntry::NativeUpdate(_) => unreachable!("attached ordinary operations"),
                }
            }
            Err(vac) => {
                if let Some(max_entities) = limits.max_entities {
                    if entries.len() >= max_entities {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::CapacityExhausted(
                                "Command buffer entity limit exceeded".into(),
                            ));
                        }
                        return self;
                    }
                }
                let mut patch = EntityPatch::new();
                if let Err(e) = op(&mut patch, pool) {
                    let mapped_err = match e {
                        StateError::ComponentDuplicateInPatch { type_name, .. } => {
                            StateError::ComponentDuplicateInPatch {
                                entity: id,
                                type_name,
                            }
                        }
                        other => other,
                    };
                    if deferred_error.is_none() {
                        *deferred_error = Some(mapped_err);
                    }
                } else {
                    let sem = patch.semantics();
                    match sem {
                        PatchSemantics::Empty => *empty_patches += 1,
                        PatchSemantics::InPlace => *in_place_patches += 1,
                        PatchSemantics::Structural => *structural_patches += 1,
                        PatchSemantics::Conditional => *conditional_patches += 1,
                    }
                    let idx = entries.len();
                    entries.push((id, CommandEntry::Patch(patch)));
                    if let Some(vac) = vac {
                        vac.insert(idx);
                    }
                    self.entry_index = Some(idx);
                    *component_op_count = new_ops;
                }
            }
        }
        self
    }

    /// 为实体添加新组件（该实体在存储中必须尚无此组件）
    #[inline]
    pub fn add<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> &mut Self {
        self.mutate::<T, _>(|p, pool| {
            p.mutate_op_pooled(
                crate::ComponentTypeId::of::<T>(),
                std::any::type_name::<T>(),
                crate::storage::batch::PatchOp::Insert(ErasedComponentValue::new(value)),
                crate::storage::batch::PatchSemantics::Structural,
                pool,
            )
        })
    }

    /// 为实体插入新组件（`add` 的兼容别名，该实体在存储中必须尚无此组件）
    #[inline]
    pub fn insert<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> &mut Self {
        self.add(value)
    }

    /// 替换实体的已有组件（该实体在存储中必须已有此组件）
    #[inline]
    pub fn replace<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> &mut Self {
        self.mutate::<T, _>(|p, pool| {
            p.mutate_op_pooled(
                crate::ComponentTypeId::of::<T>(),
                std::any::type_name::<T>(),
                crate::storage::batch::PatchOp::Replace(ErasedComponentValue::new(value)),
                crate::storage::batch::PatchSemantics::InPlace,
                pool,
            )
        })
    }

    /// 插入或替换组件（若存在则替换，若不存在则插入）
    #[inline]
    pub fn upsert<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> &mut Self {
        self.mutate::<T, _>(|p, pool| {
            p.mutate_op_pooled(
                crate::ComponentTypeId::of::<T>(),
                std::any::type_name::<T>(),
                crate::storage::batch::PatchOp::Upsert(ErasedComponentValue::new(value)),
                crate::storage::batch::PatchSemantics::Conditional,
                pool,
            )
        })
    }

    /// 设置/更新组件（若存在则替换，若不存在则插入；同一批次内多次对同一组件调用 `set` 以最后一次为准，且支持在同一批次内 `create_empty` 后追加组件）。
    #[inline]
    pub fn set<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> &mut Self {
        if self.buffer.deferred_error.is_some() {
            return self;
        }
        let cid = crate::ComponentTypeId::of::<T>();
        let new_ops = match self.buffer.component_op_count.checked_add(1) {
            Some(ops) => ops,
            None => {
                self.buffer.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        };
        if let Some(max_ops) = self.buffer.limits.max_component_ops {
            if new_ops > max_ops {
                self.buffer.record_error(StateError::CapacityExhausted(
                    "Command buffer component operations limit exceeded".into(),
                ));
                return self;
            }
        }
        let id = self.id;
        let CommandBuffer {
            limits,
            entries,
            index,
            pool,
            deferred_error,
            component_op_count,
            in_place_patches,
            structural_patches,
            conditional_patches,
            empty_patches,
            native_values,
            ..
        } = &mut *self.buffer;
        let slot = match self.entry_index {
            Some(idx) => Ok(idx),
            None if entries.last().is_some_and(|(last_id, _)| *last_id == id) => {
                Ok(entries.len() - 1)
            }
            None if index.is_empty() && entries.last().is_none_or(|(last_id, _)| *last_id < id) => {
                Err(None)
            }
            None => {
                sync_index(entries, index);
                match index.entry(id) {
                    Entry::Occupied(occ) => Ok(*occ.get()),
                    Entry::Vacant(vac) => Err(Some(vac)),
                }
            }
        };
        match slot {
            Ok(idx) => {
                self.entry_index = Some(idx);
                let native_head = entries[idx].1.native_head();
                if native_head.is_some() && updates::contains_type(native_head, native_values, cid)
                {
                    *deferred_error = Some(StateError::ComponentDuplicateInPatch {
                        entity: id,
                        type_name: std::any::type_name::<T>().into(),
                    });
                    return self;
                }
                if let CommandEntry::NativeUpdate(head) = entries[idx].1 {
                    let mut patch = EntityPatch::new();
                    patch.native_head = Some(head);
                    entries[idx].1 = CommandEntry::Patch(patch);
                }
                match &mut entries[idx].1 {
                    CommandEntry::Delete => {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::BatchConflict(format!(
                                "Cannot modify entity {id} marked for deletion in this command buffer"
                            )));
                        }
                    }
                    CommandEntry::Create(bundle) => {
                        if let Err(e) = bundle.set_pooled(cid, ErasedComponentValue::new(value), pool) {
                            if deferred_error.is_none() {
                                *deferred_error = Some(e);
                            }
                        } else {
                            *component_op_count = new_ops;
                        }
                    }
                    CommandEntry::Patch(patch) => {
                        let old_sem = if native_head.is_some() && patch.is_empty() {
                            PatchSemantics::InPlace
                        } else {
                            patch.semantics()
                        };
                        let erased = ErasedComponentValue::new(value);
                        let res = if patch.ops().contains_key_pooled(&cid, pool) {
                            patch.set_op_pooled(
                                cid,
                                crate::storage::batch::PatchOp::Upsert(erased),
                                crate::storage::batch::PatchSemantics::Conditional,
                                pool,
                            )
                        } else {
                            patch.mutate_op_pooled(
                                cid,
                                std::any::type_name::<T>(),
                                crate::storage::batch::PatchOp::Upsert(erased),
                                crate::storage::batch::PatchSemantics::Conditional,
                                pool,
                            )
                        };
                        if let Err(e) = res {
                            if deferred_error.is_none() {
                                *deferred_error = Some(e);
                            }
                        } else {
                            *component_op_count = new_ops;
                            let new_sem = patch.semantics();
                            if old_sem != new_sem {
                                match old_sem {
                                    PatchSemantics::Empty => *empty_patches -= 1,
                                    PatchSemantics::InPlace => *in_place_patches -= 1,
                                    PatchSemantics::Structural => *structural_patches -= 1,
                                    PatchSemantics::Conditional => *conditional_patches -= 1,
                                }
                                match new_sem {
                                    PatchSemantics::Empty => *empty_patches += 1,
                                    PatchSemantics::InPlace => *in_place_patches += 1,
                                    PatchSemantics::Structural => *structural_patches += 1,
                                    PatchSemantics::Conditional => *conditional_patches += 1,
                                }
                            }
                        }
                    }
                    CommandEntry::NativeUpdate(_) => unreachable!("attached ordinary operations"),
                }
            }
            Err(vac) => {
                if let Some(max_entities) = limits.max_entities {
                    if entries.len() >= max_entities {
                        if deferred_error.is_none() {
                            *deferred_error = Some(StateError::CapacityExhausted(
                                "Command buffer entity limit exceeded".into(),
                            ));
                        }
                        return self;
                    }
                }
                let mut patch = EntityPatch::new();
                if let Err(e) = patch.mutate_op_pooled(
                    cid,
                    std::any::type_name::<T>(),
                    crate::storage::batch::PatchOp::Upsert(ErasedComponentValue::new(value)),
                    crate::storage::batch::PatchSemantics::Conditional,
                    pool,
                ) {
                    if deferred_error.is_none() {
                        *deferred_error = Some(e);
                    }
                } else {
                    let sem = patch.semantics();
                    match sem {
                        PatchSemantics::Empty => *empty_patches += 1,
                        PatchSemantics::InPlace => *in_place_patches += 1,
                        PatchSemantics::Structural => *structural_patches += 1,
                        PatchSemantics::Conditional => *conditional_patches += 1,
                    }
                    let idx = entries.len();
                    entries.push((id, CommandEntry::Patch(patch)));
                    if let Some(vac) = vac {
                        vac.insert(idx);
                    }
                    self.entry_index = Some(idx);
                    *component_op_count = new_ops;
                }
            }
        }
        self
    }

    /// 移除实体的组件（该实体在存储中必须已有此组件）
    #[inline]
    pub fn remove<T: ModelComponent + 'static>(&mut self) -> &mut Self {
        self.mutate::<T, _>(|p, pool| {
            p.mutate_op_pooled(
                crate::ComponentTypeId::of::<T>(),
                std::any::type_name::<T>(),
                crate::storage::batch::PatchOp::Remove(crate::ComponentTypeId::of::<T>()),
                crate::storage::batch::PatchSemantics::Structural,
                pool,
            )
        })
    }

    /// 排队设置指向 `target` 的关系对 `Pair<R, EntityId>`。
    #[inline]
    pub fn set_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        relation: R,
        target: EntityId,
    ) -> &mut Self {
        self.set(crate::al::Pair::new(relation, target))
    }

    /// 排队移除关系对 `Pair<R, EntityId>`。
    #[inline]
    pub fn remove_pair<R: ModelComponent + 'static>(&mut self) -> &mut Self {
        self.remove::<crate::al::Pair<R, EntityId>>()
    }

    /// 排队添加零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    pub fn add_tag<const N: u8>(&mut self) -> &mut Self {
        self.set(crate::al::Tag::<N>)
    }

    /// 排队移除零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    pub fn remove_tag<const N: u8>(&mut self) -> &mut Self {
        self.remove::<crate::al::Tag<N>>()
    }

    /// 标记销毁该实体
    #[inline]
    pub fn despawn(&mut self) -> &mut Self {
        self.buffer.delete_entity(self.id);
        self
    }

    /// 标记删除该实体（`despawn` 的兼容别名）
    #[inline]
    pub fn delete(&mut self) -> &mut Self {
        self.despawn()
    }
}

#[cfg(test)]
mod tests;
