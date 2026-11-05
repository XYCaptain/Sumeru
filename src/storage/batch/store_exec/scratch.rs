use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::allocator::EntityLocation;
use crate::storage::archetype::Archetype;
use crate::storage::batch::{
    BatchOpCounts, EntityBundle, EntityPatch, NativeUpdateRef, PoolEntry,
};
use crate::storage::column::AnyColumn;
use crate::storage::core::column::PlannedInPlacePatch;
use crate::storage::core::entity_index::EntityIndex;
use crate::storage::ErasedComponentValue;
use crate::EntityId;

#[derive(Clone, Copy)]
pub(crate) struct NativeType {
    pub type_id: ComponentTypeId,
    pub rust_type_id: std::any::TypeId,
    pub name: &'static str,
}

/// 裸函数指针段级直接分发操作表（消除内层循环每实体每组件的虚表分发）
#[derive(Copy, Clone)]
pub(crate) struct NativeDirectOps {
    pub(crate) col_ptr: *mut (),
    pub(crate) rows_ptr: *mut (),
    pub(crate) type_id: ComponentTypeId,
    pub(crate) assign_fn:
        unsafe fn(col_ptr: *mut (), rows_ptr: *mut (), value_row: usize, target_row: usize),
    pub(crate) replace_fn: unsafe fn(
        col_ptr: *mut (),
        rows_ptr: *mut (),
        value_row: usize,
        target_row: usize,
        type_id: ComponentTypeId,
    ) -> ErasedComponentValue,
}

impl NativeDirectOps {
    #[inline(always)]
    pub(crate) unsafe fn assign(&self, value_row: usize, target_row: usize) {
        unsafe {
            (self.assign_fn)(self.col_ptr, self.rows_ptr, value_row, target_row);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn replace(
        &self,
        value_row: usize,
        target_row: usize,
    ) -> ErasedComponentValue {
        unsafe {
            (self.replace_fn)(
                self.col_ptr,
                self.rows_ptr,
                value_row,
                target_row,
                self.type_id,
            )
        }
    }
}

pub(crate) trait NativeLane: std::fmt::Debug + Send + Sync + 'static {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn clear_for_reuse(&mut self);
    fn cached_bytes(&self) -> usize;
    fn metadata(&self) -> NativeType;
    fn empty_column(&self) -> Box<dyn AnyColumn>;
    fn next(&self, row: usize) -> Option<NativeUpdateRef>;
    fn take(&mut self, row: usize) -> (ErasedComponentValue, Option<NativeUpdateRef>);
    fn try_plan_single(
        &mut self,
        row: usize,
        entity: EntityId,
        location: EntityLocation,
        archetype: &Archetype,
        receipt_index: usize,
    ) -> Result<bool, StateError>;
    fn planned_len(&self) -> usize;
    fn planned_archetypes(&self) -> &[u32];
    fn plan_columnar_member(
        &mut self,
        row: usize,
        entity: EntityId,
        location: EntityLocation,
        archetype: &Archetype,
        receipt_index: usize,
    ) -> Result<(), StateError>;
    fn plan_linked(
        &mut self,
        value_row: usize,
        entity: EntityId,
        location: EntityLocation,
        archetype: &Archetype,
    ) -> Result<(ComponentTypeId, Option<NativeUpdateRef>), StateError>;
    fn replace_linked(
        &mut self,
        value_row: usize,
        row: usize,
        archetype: &mut Archetype,
    ) -> ErasedComponentValue;
    fn assign_linked(&mut self, value_row: usize, row: usize, archetype: &mut Archetype);
    fn commit_discard(&mut self, archetypes: &mut [Archetype]);

    /// 预绑定：从目标 Archetype 的目标列获取可变切片裸指针，缓存到 Lane 内部。
    fn bind_linked_column(&mut self, archetype: &mut Archetype);

    /// 清除预绑定的裸指针。
    fn unbind_linked_column(&mut self);

    /// 获取段级直接分发操作表。调用前必须已完成目标列的借用与绑定。
    fn direct_ops(&mut self) -> NativeDirectOps;

    /// 标记该 Lane 的静态类型已在 Prepare 入口处完成全局注册表核验。
    fn mark_type_verified(&mut self) {}
}

/// 事务性延迟命令执行项
#[derive(Debug)]
pub(crate) enum CommandEntry {
    Create(EntityBundle),
    Patch(EntityPatch),
    Delete,
    /// Typed values remain in their lane until execution-plan construction.
    NativeUpdate(NativeUpdateRef),
}

impl CommandEntry {
    #[inline]
    pub(crate) fn for_each_value<'a>(&'a self, mut f: impl FnMut(&'a ErasedComponentValue)) {
        match self {
            Self::Create(bundle) => {
                for v in bundle.components().values() {
                    f(v);
                }
            }
            Self::Patch(patch) => {
                patch.for_each_value(f);
            }
            Self::Delete => {}
            Self::NativeUpdate(_) => {}
        }
    }

    #[inline]
    pub(crate) fn native_head(&self) -> Option<NativeUpdateRef> {
        match self {
            Self::NativeUpdate(head) => Some(*head),
            Self::Patch(patch) => patch.native_head,
            _ => None,
        }
    }
}

/// Owns the shared plan arrays for ordinary multi-entity and linked native
/// updates. Singleton ordinary and standalone-native plans stay lightweight.
#[derive(Debug, Default)]
pub(crate) struct PlanScratch {
    pub(crate) in_place_plans: Vec<PlannedInPlacePatch>,
    pub(crate) native_references: Vec<(crate::ComponentTypeId, NativeUpdateRef)>,
}

/// Owns native lanes across recording, preparation, cancellation and commit.
#[derive(Debug, Default)]
pub(crate) struct NativeLanes {
    lanes: Vec<Box<dyn NativeLane>>,
}

impl std::ops::Deref for NativeLanes {
    type Target = Vec<Box<dyn NativeLane>>;
    fn deref(&self) -> &Self::Target {
        &self.lanes
    }
}
impl std::ops::DerefMut for NativeLanes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lanes
    }
}
impl<'a> IntoIterator for &'a NativeLanes {
    type Item = &'a Box<dyn NativeLane>;
    type IntoIter = std::slice::Iter<'a, Box<dyn NativeLane>>;
    fn into_iter(self) -> Self::IntoIter {
        self.lanes.iter()
    }
}

#[derive(Debug)]
pub(crate) struct CommandScratch {
    pub entries: Vec<(EntityId, CommandEntry)>,
    pub index: EntityIndex<usize>,
    pub pool: Vec<PoolEntry>,
}

impl CommandScratch {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: EntityIndex::default(),
            pool: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_capacity(entities: usize, ops: usize) -> Self {
        Self {
            entries: Vec::with_capacity(entities),
            index: EntityIndex::with_capacity_and_hasher(entities, Default::default()),
            pool: Vec::with_capacity(ops),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.pool.clear();
    }
}

/// 包装条目与精确类别计数的内部输入容器，避免准备阶段重复统计
pub(crate) struct CommandEntries {
    pub(crate) entries: Vec<(EntityId, CommandEntry)>,
    pub(crate) index: EntityIndex<usize>,
    pub(crate) pool: Vec<PoolEntry>,
    pub(crate) native_values: NativeLanes,
    pub(crate) counts: BatchOpCounts,
}
