//! StateFabric 统一列存数据内核 (Store)

pub(crate) mod crud;
pub(crate) mod despawn;
pub(crate) mod direct;
pub(crate) mod registration;
pub(crate) mod snapshot;

#[allow(unused_imports)]
pub(crate) use crud::ReplaceInPlaceResult;
pub use crud::Singleton;
pub use direct::{TypedBatch, TypedDirectBatchUpdate, TypedReceipt};
pub use snapshot::{ColumnarStoreSnapshot, CURRENT_SNAPSHOT_VERSION};

use crate::storage::allocator::{EntityAllocator, EntityLocation};
use crate::storage::archetype::Archetype;
use crate::error::StateError;
use crate::EntityId;
#[cfg(any(test, feature = "test-internals"))]
use crate::ModelComponent;
use crate::id::ComponentTypeId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 列存储物理布局优化策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LayoutOptimization {
    /// 紧凑连续内存布局（默认）
    #[default]
    Compact,
    /// 快速写入优先布局
    FastWrite,
    /// 16KB 硬件页对齐零拷贝 GPU 物理布局（Apple Metal UMA / Vulkan / WebGPU）
    PageAlignedGpu,
}

static NEXT_STORE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
pub(crate) fn next_store_instance_id() -> u64 {
    NEXT_STORE_INSTANCE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("Store instance identity exhausted")
}

/// 组件所属 Archetype 的倒排位图索引（每 64 个 Archetype 1 个 `u64` 字）
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ArchetypeBitSet {
    words: smallvec::SmallVec<[u64; 2]>,
}

impl ArchetypeBitSet {
    #[inline]
    pub(crate) fn insert(&mut self, arch_id: u32) {
        let idx = (arch_id as usize) >> 6;
        let bit = (arch_id as usize) & 63;
        if idx >= self.words.len() {
            self.words.resize(idx + 1, 0);
        }
        self.words[idx] |= 1u64 << bit;
    }

    #[inline(always)]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub(crate) fn intersect_with(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        self.words.truncate(min_len);
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }
    }

    #[inline]
    pub(crate) fn difference_with(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= !other.words[i];
        }
    }

    #[inline]
    pub(crate) fn is_all_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    #[inline]
    pub(crate) fn for_each_set_bit(&self, mut f: impl FnMut(u32)) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            let base = (word_idx as u32) << 6;
            while w != 0 {
                let bit = w.trailing_zeros();
                f(base + bit);
                w &= w - 1;
            }
        }
    }
}

/// 统一列式状态存储核心 (Store)。
///
/// 具备代际防 ABA、Archetype 动态组件布局、严格单所有者写控制与只读 Query 访问能力。
pub struct Store {
    pub(crate) store_id: u64,
    pub(crate) epoch: u64,
    pub(crate) archetype_set_generation: u64,
    pub(crate) archetype_topology_epoch: u64,
    pub(crate) allocator: EntityAllocator,
    pub(crate) archetypes: Vec<Archetype>,
    pub(crate) archetype_index: BTreeMap<BTreeSet<ComponentTypeId>, u32>,
    pub(crate) component_archetypes: BTreeMap<ComponentTypeId, ArchetypeBitSet>,
    pub(crate) last_created_archetype: Option<u32>,
    pub(crate) component_registry: crate::storage::registry::ComponentRegistry,
    pub(crate) poisoned: Arc<AtomicBool>,
    pub(crate) layout_optimization: LayoutOptimization,
    pub(crate) structural_validation_cache: std::sync::RwLock<StructuralValidationCache>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct ArchetypeValidationBitmap {
    lo: u64,
    hi: u64,
    overflow: Option<BTreeSet<u32>>,
}

impl ArchetypeValidationBitmap {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn contains(&self, arch_id: u32) -> bool {
        if arch_id < 64 {
            (self.lo & (1u64 << arch_id)) != 0
        } else if arch_id < 128 {
            (self.hi & (1u64 << (arch_id - 64))) != 0
        } else {
            self.overflow.as_ref().is_some_and(|set| set.contains(&arch_id))
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, arch_id: u32) -> bool {
        if arch_id < 64 {
            let mask = 1u64 << arch_id;
            if (self.lo & mask) != 0 {
                false
            } else {
                self.lo |= mask;
                true
            }
        } else if arch_id < 128 {
            let mask = 1u64 << (arch_id - 64);
            if (self.hi & mask) != 0 {
                false
            } else {
                self.hi |= mask;
                true
            }
        } else {
            self.overflow
                .get_or_insert_with(BTreeSet::new)
                .insert(arch_id)
        }
    }
}


/// Archetype 结构性校验缓存。
/// 记录已通过结构性校验（ID与索引一致、反查一致、列键集合一致、类型名称一致等不变量）的 Archetype ID。
/// 动态行数对齐始终在运行时实时检查，不进行跳过。
#[derive(Debug, Default, Clone)]
pub(crate) struct StructuralValidationCache {
    pub(crate) store_id: u64,
    pub(crate) generation: u64,
    pub(crate) validated: ArchetypeValidationBitmap,
}

impl StructuralValidationCache {
    #[inline]
    pub(crate) fn is_valid(&self, current_gen: u64) -> bool {
        self.generation == current_gen
    }

    #[inline]
    pub(crate) fn reset(&mut self, current_gen: u64) {
        self.generation = current_gen;
        self.validated = ArchetypeValidationBitmap::new();
    }
}

/// 物理提交过程异常守卫：若执行物理变更期间发生 panic（如用户析构异常），自动将 Store 标记为 Panic 故障污染状态（Poisoned）。
pub(crate) struct CorePoisonGuard {
    poisoned: *const AtomicBool,
    armed: bool,
}

impl CorePoisonGuard {
    #[inline]
    pub(crate) fn new(poisoned: &AtomicBool) -> Self {
        Self {
            poisoned: poisoned as *const AtomicBool,
            armed: true,
        }
    }

    #[inline]
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CorePoisonGuard {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            unsafe {
                (*self.poisoned).store(true, Ordering::Release);
            }
        }
    }
}



impl Store {
    /// 原生存储后端标识常量。
    pub const BACKEND_NAME: &'static str = "ens_store";

    /// Check a sequence of writes before a domain-level undo begins.
    pub fn validate_write_count(&self, count: u64) -> Result<(), StateError> {
        self.epoch
            .checked_add(count)
            .map(|_| ())
            .ok_or_else(|| StateError::CapacityExhausted("Store revision exhausted".into()))
    }

    /// Explicit diagnostic check; not performed on every read or replacement.
    pub fn validate_invariants(&self) -> Result<(), StateError> {
        self.allocator.validate()?;
        if self.archetypes.len() != self.archetype_index.len() {
            return Err(StateError::AccessDenied(
                "Archetype index size mismatch".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for (index, arch) in self.archetypes.iter().enumerate() {
            self.validate_archetype_layout_readonly(index)?;
            for (row, &entity) in arch.entities().iter().enumerate() {
                if !seen.insert(entity)
                    || self.allocator.get_location(entity)
                        != Some(EntityLocation::new(index as u32, row as u32))
                {
                    return Err(StateError::AccessDenied(
                        "Entity row mapping mismatch".into(),
                    ));
                }
            }
        }
        if seen.len() != self.len() {
            return Err(StateError::AccessDenied(
                "Allocator contains an orphan entity".into(),
            ));
        }
        Ok(())
    }

    /// 全局唯一的存储库实例标识符
    #[inline]
    pub fn store_id(&self) -> u64 {
        self.store_id
    }

    /// 单调递增的写代号（用于防过期提交与快照失效保护）
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// 将所有 Archetype 列的归档编码器 (Codec) 与 16KB 页对齐布局规范化至当前 Store 的注册表与布局策略。
    pub(crate) fn normalize_archetypes_to_registry(&mut self, stamp_epoch: Option<u64>) -> usize {
        if self.layout_optimization == LayoutOptimization::PageAlignedGpu {
            self.component_registry.promote_all_to_page_aligned();
        }
        let page_aligned_mode = self.layout_optimization == LayoutOptimization::PageAlignedGpu;
        let mut promoted = 0usize;
        for arch in &mut self.archetypes {
            if let Some(epoch) = stamp_epoch {
                arch.stamp_all_rows_epoch(epoch);
            }
            for (&cid, col) in arch.columns_mut().iter_mut() {
                let mut should_promote = page_aligned_mode;
                if let Some(reg_def) = self.component_registry.get(&cid) {
                    col.upgrade_archive_capability_from(reg_def.prototype());
                    if reg_def.prototype().is_page_aligned() {
                        should_promote = true;
                    }
                }
                if should_promote && !col.is_page_aligned() {
                    col.promote_to_page_aligned();
                    promoted += 1;
                }
            }
            arch.rebuild_flat_columns();
        }
        promoted
    }

    /// 配置列存储物理布局优化策略
    #[inline]
    pub fn with_layout_optimization(mut self, opt: LayoutOptimization) -> Self {
        self.layout_optimization = opt;
        if opt == LayoutOptimization::PageAlignedGpu {
            self.normalize_archetypes_to_registry(None);
        }
        self
    }

    /// 获取当前列存储物理布局优化策略
    #[inline]
    pub fn layout_optimization(&self) -> LayoutOptimization {
        self.layout_optimization
    }

    /// 执行物理布局整理（在 `PageAlignedGpu` 模式下将未对齐列提升为 16KB 页对齐，返回提升列数）
    #[inline]
    pub fn optimize_layout(&mut self) -> usize {
        if self.layout_optimization != LayoutOptimization::PageAlignedGpu {
            return 0;
        }
        self.normalize_archetypes_to_registry(None)
    }

    #[inline]
    pub(crate) fn next_epoch(&self) -> Result<u64, StateError> {
        self.epoch
            .checked_add(1)
            .ok_or_else(|| StateError::CapacityExhausted("Store revision exhausted".into()))
    }

    pub(crate) fn bump_epoch(&mut self) {
        self.epoch = self
            .next_epoch()
            .expect("revision must be checked before writing");
    }

    #[cfg(any(test, feature = "test-internals"))]
    pub fn advance_epoch_for_test(&mut self, delta: u64) {
        self.epoch = self.epoch.checked_add(delta).expect("revision must not overflow");
    }

    #[cfg(any(test, feature = "test-internals"))]
    pub fn set_epoch_for_test(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Archetype 表集合版本（表集合变更时递增，用于查询计划缓存失效）
    #[inline]
    pub fn archetype_set_generation(&self) -> u64 {
        self.archetype_set_generation
    }

    /// 原型拓扑时代纪元（仅在快照还原、清空重建等只增不减假设被打破时递增）
    #[inline]
    pub(crate) fn archetype_topology_epoch(&self) -> u64 {
        self.archetype_topology_epoch
    }

    /// 组件类型到包含该组件的 Archetype 位图倒排索引
    #[inline]
    pub(crate) fn component_archetypes(&self) -> &BTreeMap<ComponentTypeId, ArchetypeBitSet> {
        &self.component_archetypes
    }

    /// 将新创建的 Archetype 注册到组件倒排位图索引中
    #[inline]
    pub(crate) fn register_archetype_in_component_index(
        &mut self,
        arch_id: u32,
        component_types: &BTreeSet<ComponentTypeId>,
    ) {
        for &cid in component_types {
            self.component_archetypes
                .entry(cid)
                .or_default()
                .insert(arch_id);
        }
    }

    /// 在快照恢复或拓扑重置后全量重建组件倒排位图索引并递增拓扑纪元
    pub(crate) fn rebuild_component_archetypes_on_topology_reset(&mut self) {
        self.component_archetypes.clear();
        for arch in &self.archetypes {
            let arch_id = arch.id().0;
            for &cid in arch.component_types() {
                self.component_archetypes
                    .entry(cid)
                    .or_default()
                    .insert(arch_id);
            }
        }
        self.archetype_topology_epoch = self.archetype_topology_epoch.wrapping_add(1);
    }

    /// 设置实体在当前原型行上的非迁移状态位（`0..=255`），不触发跨表迁移，返回修改前的旧状态值
    #[inline]
    pub fn set_flag<const FLAG: u8>(
        &mut self,
        entity: EntityId,
        enabled: bool,
    ) -> Result<bool, StateError> {
        self.check_poisoned()?;
        let loc = self
            .allocator
            .get_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        let arch = self
            .archetypes
            .get_mut(loc.archetype_id as usize)
            .ok_or(StateError::EntityNotFound(entity))?;
        Ok(arch.set_row_flag(loc.row_index as usize, FLAG, enabled))
    }

    /// 查询实体在当前原型行上的非迁移状态位（`0..=255`）是否已置位
    #[inline]
    pub fn has_flag<const FLAG: u8>(&self, entity: EntityId) -> bool {
        let Some(loc) = self.allocator.get_location(entity) else {
            return false;
        };
        let Some(arch) = self.archetypes.get(loc.archetype_id as usize) else {
            return false;
        };
        arch.get_row_flag(loc.row_index as usize, FLAG)
    }

    #[inline]
    pub(crate) fn check_archetype_set_generation_capacity(&self) -> Result<(), StateError> {
        self.archetype_set_generation
            .checked_add(1)
            .ok_or(StateError::ArchetypeGenerationOverflow)?;
        Ok(())
    }

    #[inline]
    pub(crate) fn bump_archetype_set_generation(&mut self) -> Result<(), StateError> {
        self.check_archetype_set_generation_capacity()?;
        self.archetype_set_generation = self
            .archetype_set_generation
            .checked_add(1)
            .ok_or(StateError::ArchetypeGenerationOverflow)?;
        // 结构变更时代际递增，使持久的结构性校验缓存失效
        self.structural_validation_cache.write().unwrap().reset(self.archetype_set_generation);
        Ok(())
    }

    /// 查询当前存储核心是否已处于 Panic 故障污染状态（Poisoned）
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// 测试与故障注入专用：手动将存储核心置为 Panic 故障污染状态（Poisoned）
    #[doc(hidden)]
    pub fn set_poisoned_for_test(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// 测试专用：获取指定实体所在组件列的分段数量
    #[cfg(any(test, feature = "test-internals"))]
    pub fn column_segment_count<T: ModelComponent + 'static>(&self, entity: EntityId) -> Option<usize> {
        let loc = self.allocator.get_location(entity)?;
        let arch = self.archetypes.get(loc.archetype_id as usize)?;
        let cid = ComponentTypeId::of::<T>();
        let col = arch.columns().get(&cid)?;
        Some(col.segment_count())
    }

    /// 测试专用：检查指定实体所在组件列是否处于分段或共享冻结态
    #[cfg(any(test, feature = "test-internals"))]
    pub fn is_column_segmented<T: ModelComponent + 'static>(&self, entity: EntityId) -> Option<bool> {
        let loc = self.allocator.get_location(entity)?;
        let arch = self.archetypes.get(loc.archetype_id as usize)?;
        let cid = ComponentTypeId::of::<T>();
        let col = arch.columns().get(&cid)?;
        Some(col.is_segmented())
    }

    /// 校验存储核心未处于 Panic 故障污染状态，若已污染返回 SessionPoisoned 错误
    #[inline]
    pub(crate) fn check_poisoned(&self) -> Result<(), StateError> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(StateError::SessionPoisoned)
        } else {
            Ok(())
        }
    }

    /// 按 Archetype 批量迭代实体切片及其组件类型集合
    #[inline]
    pub fn component_sets(
        &self,
    ) -> impl Iterator<Item = (&[EntityId], &BTreeSet<ComponentTypeId>)> + '_ {
        self.archetypes
            .iter()
            .map(|arch| (arch.entities(), arch.component_types()))
    }

    pub fn new() -> Self {
        let store_id = next_store_instance_id();
        let mut store = Self {
            store_id,
            epoch: 1,
            archetype_set_generation: 1,
            archetype_topology_epoch: 1,
            allocator: EntityAllocator::new(),
            archetypes: Vec::new(),
            archetype_index: BTreeMap::new(),
            component_archetypes: BTreeMap::new(),
            last_created_archetype: None,
            component_registry: crate::storage::registry::ComponentRegistry::new(),
            poisoned: Arc::new(AtomicBool::new(false)),
            layout_optimization: LayoutOptimization::default(),
            structural_validation_cache: std::sync::RwLock::new(StructuralValidationCache {
                store_id,
                generation: 0,
                validated: ArchetypeValidationBitmap::new(),
            }),
        };
        // 预创建空组件 Archetype 0
        store.get_or_create_archetype(BTreeSet::new());
        store
    }

    /// 创建预留指定实体分配容量的存储核心。
    pub fn with_capacity(entity_capacity: usize) -> Self {
        let mut store = Self::new();
        if entity_capacity > 0 {
            let _ = store.reserve_entities(entity_capacity);
        }
        store
    }

    /// 返回当前包含至少 1 个活跃实体的非空原型表数量。
    #[inline]
    pub fn populated_archetype_count(&self) -> usize {
        self.archetypes.iter().filter(|a| !a.is_empty()).count()
    }

    /// 派生具有相同原型拓扑与组件注册表、但实体为空的独立分区副本（`empty_with_schema` 的语义化门面别名）。
    #[inline]
    pub fn fork_empty(&self) -> Self {
        self.empty_with_schema()
    }

    /// 检查另一个独立分区 `Store` 是否与当前 `Store` 实体完全不相交且组件定义兼容（`can_merge_disjoint` 的语义化门面别名）。
    #[inline]
    pub fn can_merge_partition(&self, other: &Self) -> bool {
        self.can_merge_disjoint(other)
    }

    /// 将另一个实体集合互不相交的独立分区 `Store` 批量合并至当前 `Store`（`merge_disjoint_from` 的语义化门面别名）。
    #[inline]
    pub fn merge_partition(&mut self, other: Self) -> Result<(), StateError> {
        self.merge_disjoint_from(other)
    }

    /// 只读全量校验（用于诊断断言），不修改缓存。
    pub(crate) fn validate_archetype_layout_readonly(&self, index: usize) -> Result<(), StateError> {
        let archetype = self
            .archetypes
            .get(index)
            .ok_or_else(|| StateError::AccessDenied(format!("Unknown archetype {index}")))?;
        if archetype.id().0 as usize != index
            || self.archetype_index.get(archetype.component_types()) != Some(&archetype.id().0)
            || !archetype
                .columns()
                .keys()
                .eq(archetype.component_types().iter())
            || archetype.columns().iter().any(|(&id, column)| {
                column.len() != archetype.len()
                    || column.as_ref().type_id() != id
                    || ComponentTypeId::from_name(column.type_name()) != id
                    || column.rust_type_id().is_none()
            })
        {
            return Err(StateError::AccessDenied(format!(
                "Archetype {index} has inconsistent native column metadata or layout"
            )));
        }
        Ok(())
    }

    /// 布局校验：结构性常量（ID、索引、列键集合、类型映射）通过缓存跨批次持久复用；
    /// 动态行数对齐始终在运行时实时检查，保证行安全。
    pub(crate) fn validate_archetype_layout(&self, index: usize) -> Result<(), StateError> {
        if self.archetype_index.len() != self.archetypes.len() {
            return Err(StateError::AccessDenied(format!(
                "Archetype {index} has inconsistent native column metadata or layout"
            )));
        }
        let current_gen = self.archetype_set_generation;
        let is_structurally_validated = if let Ok(cache) = self.structural_validation_cache.read() {
            cache.is_valid(current_gen) && cache.validated.contains(index as u32)
        } else {
            false
        };
        
        let archetype = self
            .archetypes
            .get(index)
            .ok_or_else(|| StateError::AccessDenied(format!("Unknown archetype {index}")))?;

        if !is_structurally_validated {
            // 首次校验该 Archetype：执行静态结构不变量校验（①②③⑤⑥⑦）
            if archetype.id().0 as usize != index
                || self.archetype_index.get(archetype.component_types()) != Some(&archetype.id().0)
                || !archetype
                    .columns()
                    .keys()
                    .eq(archetype.component_types().iter())
                || archetype.columns().iter().any(|(&id, column)| {
                    column.as_ref().type_id() != id
                        || ComponentTypeId::from_name(column.type_name()) != id
                        || column.rust_type_id().is_none()
                })
            {
                return Err(StateError::AccessDenied(format!(
                    "Archetype {index} has inconsistent native column metadata or layout"
                )));
            }

            // 唯有在所有静态不变量校验成功且无任何错误抛出后，才安全登记到位图中！
            if let Ok(mut cache) = self.structural_validation_cache.write() {
                if !cache.is_valid(current_gen) {
                    cache.reset(current_gen);
                }
                cache.validated.insert(index as u32);
            }
        }

        // ④ 动态行数对齐检查：当原型未标记为可能分段或失步且已完成结构校验时，O(1) 短路通过；否则严格遍历全列！
        if !is_structurally_validated || archetype.may_be_segmented_or_unaligned() {
            if archetype.columns().values().any(|column| column.len() != archetype.len()) {
                return Err(StateError::AccessDenied(format!(
                    "Archetype {index} column length mismatch with entity count"
                )));
            }
        }
        Ok(())
    }

    /// 独占可变借用下的布局校验：通过 `RwLock::get_mut()` 零原子锁开销读取与更新结构性校验缓存。
    pub(crate) fn validate_archetype_layout_mut(&mut self, index: usize) -> Result<(), StateError> {
        if self.archetype_index.len() != self.archetypes.len() {
            return Err(StateError::AccessDenied(format!(
                "Archetype {index} has inconsistent native column metadata or layout"
            )));
        }
        let current_gen = self.archetype_set_generation;
        let is_structurally_validated =
            if let Ok(cache) = self.structural_validation_cache.get_mut() {
                cache.is_valid(current_gen) && cache.validated.contains(index as u32)
            } else {
                false
            };

        let archetype = self
            .archetypes
            .get(index)
            .ok_or_else(|| StateError::AccessDenied(format!("Unknown archetype {index}")))?;

        if is_structurally_validated && !archetype.may_be_segmented_or_unaligned() {
            return Ok(());
        }

        if !is_structurally_validated {
            if archetype.id().0 as usize != index
                || self.archetype_index.get(archetype.component_types()) != Some(&archetype.id().0)
                || !archetype
                    .columns()
                    .keys()
                    .eq(archetype.component_types().iter())
                || archetype.columns().iter().any(|(&id, column)| {
                    column.as_ref().type_id() != id
                        || ComponentTypeId::from_name(column.type_name()) != id
                        || column.rust_type_id().is_none()
                })
            {
                return Err(StateError::AccessDenied(format!(
                    "Archetype {index} has inconsistent native column metadata or layout"
                )));
            }

            if let Ok(cache) = self.structural_validation_cache.get_mut() {
                if !cache.is_valid(current_gen) {
                    cache.reset(current_gen);
                }
                cache.validated.insert(index as u32);
            }
        }

        let archetype = &self.archetypes[index];
        if archetype
            .columns()
            .values()
            .any(|column| column.len() != archetype.len())
        {
            return Err(StateError::AccessDenied(format!(
                "Archetype {index} column length mismatch with entity count"
            )));
        }
        self.archetypes[index].rebuild_flat_columns();
        Ok(())
    }

    /// 注册强类型组件模式定义，为物理列赋予可选的 rkyv 整列快照归档能力。
    ///

    /// 清空存储内所有实体与组件行数据，同时保留已构建的原型拓扑、转移边缓存与底层列/分配器容量。
    pub fn clear(&mut self) {
        self.allocator.clear();
        if self.archetype_index.len() != self.archetypes.len() || self.archetypes.is_empty() {
            self.archetypes.clear();
            self.archetype_index.clear();
            self.component_archetypes.clear();
            self.last_created_archetype = None;
            let _ = self.bump_archetype_set_generation();
            self.get_or_create_archetype(BTreeSet::new());
            self.rebuild_component_archetypes_on_topology_reset();
        } else {
            for arch in &mut self.archetypes {
                arch.clear_rows();
            }
        }
        self.epoch = self.epoch.wrapping_add(1);
    }

    pub fn empty_with_schema(&self) -> Self {
        let store_id = next_store_instance_id();
        let archetypes = self
            .archetypes
            .iter()
            .map(|source| source.empty_with_schema())
            .collect();
        Self {
            store_id,
            epoch: 1,
            archetype_set_generation: self.archetype_set_generation,
            archetype_topology_epoch: self.archetype_topology_epoch,
            allocator: EntityAllocator::new(),
            archetypes,
            archetype_index: self.archetype_index.clone(),
            component_archetypes: self.component_archetypes.clone(),
            last_created_archetype: None,
            component_registry: self.component_registry.clone(),
            poisoned: Arc::new(AtomicBool::new(false)),
            layout_optimization: self.layout_optimization,
            structural_validation_cache: std::sync::RwLock::new(StructuralValidationCache::default()),
        }
    }


    /// 获取实体的物理定位 (ArchetypeId 与行索引)
    #[inline]
    #[cfg(test)]
    pub(crate) fn get_entity_location(&self, entity: EntityId) -> Option<EntityLocation> {
        self.allocator.get_location(entity)
    }

    /// 检查指定组件类型是否已在当前 `Store` 的 `ComponentRegistry` 中注册。
    #[inline]
    pub fn has_registered_component(&self, type_id: ComponentTypeId) -> bool {
        self.component_registry.contains(&type_id)
    }

    /// 检查另一个独立解码的 `Store` 是否与当前 `Store` 实体完全不相交且组件定义兼容
    pub fn can_merge_disjoint(&self, other: &Self) -> bool {
        if self.is_poisoned() || other.is_poisoned() {
            return false;
        }
        for (&cid, def) in other.component_registry.iter() {
            if let Some(existing) = self.component_registry.get(&cid) {
                if existing.check_merge_compatibility(def.prototype()).is_err() {
                    return false;
                }
            }
        }
        if other.allocator.is_empty() {
            return other.archetypes.iter().all(|a| a.is_empty());
        }
        if self
            .allocator
            .total_slots()
            .checked_add(other.allocator.active_count())
            .is_none_or(|total| total >= u32::MAX as usize)
        {
            return false;
        }
        for other_arch in &other.archetypes {
            if other_arch.is_empty() {
                continue;
            }
            for &entity in other_arch.entities() {
                if entity.is_nil() || self.allocator.is_alive(entity) {
                    return false;
                }
            }
            if let Some(&target_arch_id) = self.archetype_index.get(other_arch.component_types()) {
                let Some(target_arch) = self.archetypes.get(target_arch_id as usize) else {
                    return false;
                };
                if target_arch
                    .len()
                    .checked_add(other_arch.len())
                    .is_none_or(|len| len > u32::MAX as usize)
                {
                    return false;
                }
                if target_arch.columns().len() != other_arch.columns().len() {
                    return false;
                }
                for (&cid, other_col) in other_arch.columns() {
                    let Some(target_col) = target_arch.columns().get(&cid) else {
                        return false;
                    };
                    if !other_col.is_empty() && other_col.validate_copy_to(0, &**target_col).is_err()
                    {
                        return false;
                    }
                }
            } else if self.archetypes.len() >= u32::MAX as usize {
                return false;
            }
        }
        true
    }

    /// 将另一个实体集合互不相交的独立 `Store` 批量合并至当前 `Store`（供多分区快照直装使用）
    pub fn merge_disjoint_from(&mut self, other: Self) -> Result<(), StateError> {
        self.check_poisoned()?;
        other.check_poisoned()?;
        if !self.can_merge_disjoint(&other) {
            return Err(StateError::AccessDenied(
                "Cannot merge overlapping or incompatible Store".into(),
            ));
        }

        let mut other_prototypes = Vec::new();
        for (&cid, def) in other.component_registry.iter() {
            if let Some(existing) = self.component_registry.get(&cid) {
                existing.check_merge_compatibility(def.prototype())?;
                if (!existing.has_archive_capability() && def.has_archive_capability())
                    || (!existing.is_page_aligned() && def.is_page_aligned())
                {
                    let mut proto = if def.has_archive_capability() {
                        def.create_empty_column()
                    } else {
                        existing.create_empty_column()
                    };
                    if existing.is_page_aligned()
                        || def.is_page_aligned()
                        || self.layout_optimization == LayoutOptimization::PageAlignedGpu
                    {
                        proto.promote_to_page_aligned();
                    }
                    other_prototypes.push(proto);
                }
            } else {
                let mut proto = def.create_empty_column();
                if self.layout_optimization == LayoutOptimization::PageAlignedGpu {
                    proto.promote_to_page_aligned();
                }
                other_prototypes.push(proto);
            }
        }
        if !other_prototypes.is_empty() {
            self.component_registry
                .register_prototypes(other_prototypes)?;
            self.normalize_archetypes_to_registry(None);
        }

        if other.allocator.is_empty() {
            return Ok(());
        }

        let merge_epoch = self
            .epoch
            .max(other.epoch)
            .checked_add(1)
            .ok_or_else(|| StateError::CapacityExhausted("Store revision exhausted".into()))?;
        self.check_archetype_set_generation_capacity()?;

        if self.allocator.is_empty()
            && self.allocator.total_slots() == 0
            && self.archetypes.iter().all(|a| a.is_empty())
        {
            let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
            self.allocator = other.allocator;
            self.archetypes = other.archetypes;
            self.archetype_index = other.archetype_index;
            self.last_created_archetype = None;
            self.normalize_archetypes_to_registry(Some(merge_epoch));
            self.rebuild_component_archetypes_on_topology_reset();
            self.epoch = merge_epoch;
            self.bump_archetype_set_generation()?;
            poison_guard.disarm();
            return Ok(());
        }

        self.allocator
            .prepare_allocations(other.allocator.active_count())?;

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let mut added_archetypes = false;

        for mut other_arch in other.archetypes {
            if other_arch.is_empty() {
                continue;
            }
            if let Some(&target_arch_id) = self.archetype_index.get(other_arch.component_types()) {
                let target_arch = &mut self.archetypes[target_arch_id as usize];
                let base_row = target_arch.len() as u32;
                let other_len = other_arch.len();
                for (cid, mut other_col) in std::mem::take(other_arch.columns_mut()) {
                    let target_col = target_arch
                        .columns_mut()
                        .get_mut(&cid)
                        .expect("validated target column");
                    if let Some(existing_def) = self.component_registry.get(&cid) {
                        target_col.upgrade_archive_capability_from(existing_def.prototype());
                    }
                    if let Some(push_range) = other_col.shared_push_clone_range_raw() {
                        let (ptr, len, _writable) = other_col.access_parts();
                        target_col.reserve(len);
                        if len > 0 {
                            unsafe {
                                push_range(ptr as *const u8, 0, len, &mut **target_col)?;
                            }
                        }
                    } else {
                        target_col.reserve(other_len);
                        for row in 0..other_len {
                            other_col.copy_row_to_infallible(row, &mut **target_col);
                        }
                    }
                }
                target_arch
                    .entities_mut()
                    .extend_from_slice(other_arch.entities());
                let start_row = base_row as usize;
                for row in 0..other_len {
                    other_arch.copy_row_flags_to(row, target_arch, start_row + row);
                }
                target_arch.mark_all_columns_row_range_epoch(
                    start_row..(start_row + other_len),
                    merge_epoch,
                );
                target_arch.rebuild_flat_columns();
                self.allocator.allocate_batch_with_contiguous_rows(
                    other_arch.entities(),
                    target_arch_id,
                    base_row,
                )?;
            } else {
                let new_arch_id = u32::try_from(self.archetypes.len()).map_err(|_| {
                    StateError::CapacityExhausted("Archetype capacity exceeds u32".into())
                })?;
                self.allocator.allocate_batch_with_contiguous_rows(
                    other_arch.entities(),
                    new_arch_id,
                    0,
                )?;
                let comp_types = other_arch.component_types().clone();
                let mut new_arch = Archetype::new(
                    crate::id::ArchetypeId(new_arch_id),
                    comp_types.clone(),
                );
                *new_arch.entities_mut() = std::mem::take(other_arch.entities_mut());
                *new_arch.columns_mut() = std::mem::take(other_arch.columns_mut());
                new_arch.set_flag_planes(other_arch.take_flag_planes());
                new_arch.stamp_all_rows_epoch(merge_epoch);
                new_arch.rebuild_flat_columns();
                self.archetypes.push(new_arch);
                self.register_archetype_in_component_index(new_arch_id, &comp_types);
                self.archetype_index.insert(comp_types, new_arch_id);
                added_archetypes = true;
            }
        }

        self.normalize_archetypes_to_registry(None);
        self.last_created_archetype = None;
        self.epoch = merge_epoch;
        if added_archetypes {
            self.bump_archetype_set_generation()?;
        }
        poison_guard.disarm();
        Ok(())
    }
}


impl Clone for Store {
    fn clone(&self) -> Self {
        let store_id = next_store_instance_id();
        let mut cache = self.structural_validation_cache.read().unwrap().clone();
        cache.store_id = store_id;
        Self {
            store_id,
            epoch: 1,
            archetype_set_generation: self.archetype_set_generation,
            archetype_topology_epoch: self.archetype_topology_epoch,
            allocator: self.allocator.clone(),
            archetypes: self.archetypes.clone(),
            archetype_index: self.archetype_index.clone(),
            component_archetypes: self.component_archetypes.clone(),
            component_registry: self.component_registry.clone(),
            last_created_archetype: None,
            poisoned: Arc::new(AtomicBool::new(self.poisoned.load(Ordering::Relaxed))),
            layout_optimization: self.layout_optimization,
            structural_validation_cache: std::sync::RwLock::new(cache),
        }
    }
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("store_id", &self.store_id)
            .field("epoch", &self.epoch)
            .field("archetype_set_generation", &self.archetype_set_generation)
            .field("active_entities", &self.allocator.active_count())
            .field("archetypes_count", &self.archetypes.len())
            .finish()
    }
}

impl Serialize for Store {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let snapshot = self.snapshot().map_err(serde::ser::Error::custom)?;
        snapshot.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Store {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let snapshot = ColumnarStoreSnapshot::deserialize(deserializer)?;
        let mut store = Store::new();
        store
            .restore_snapshot(&snapshot)
            .map_err(serde::de::Error::custom)?;
        Ok(store)
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}


