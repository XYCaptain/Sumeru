//! Archetype 原型表管理与 Swap-Remove 同步 (Archetype)

use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ArchetypeId, ComponentTypeId, LocalSlotId};
use crate::storage::column::{AnyColumn, TypedColumn};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet};

/// 与 `component_types_slice` 1:1 对齐的扁平列指针缓存项（消除 BTreeMap 节点追逐与虚表 downcast）
#[derive(Clone, Copy)]
pub(crate) struct FlatColumnEntry {
    pub type_id: ComponentTypeId,
    pub rust_type: Option<std::any::TypeId>,
    pub col_ptr: *mut dyn AnyColumn,
    pub move_row_fn: unsafe fn(*mut (), usize, *mut ()),
    pub swap_remove_fn: unsafe fn(*mut (), usize),
    pub push_erased_fn: unsafe fn(*mut (), &mut crate::storage::ErasedComponentValue),
}

// SAFETY: `col_ptr` 指向当前 `Archetype` 自身 `columns` 持有的列实例。
// 只读访问（`&Archetype`）仅将其视作不可变引用，可变访问要求持有 `&mut Archetype`。
unsafe impl Send for FlatColumnEntry {}
unsafe impl Sync for FlatColumnEntry {}

/// 行级位图掩码（每 64 行 1 个 `u64` 字），用于查询过滤的 SIMD/字级位与交集与 `trailing_zeros` 跳跃迭代
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowBitMask {
    words: smallvec::SmallVec<[u64; 4]>,
    row_len: usize,
}

impl RowBitMask {
    /// 创建长度为 `row_len`、所有位均置 0 的空位图
    #[inline]
    pub fn new_empty(row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        Self {
            words: smallvec::smallvec![0u64; num_words],
            row_len,
        }
    }

    /// 创建长度为 `row_len`、所有有效行位均置 1 的满位图
    #[inline]
    pub fn new_all_set(row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        let mut words = smallvec::smallvec![!0u64; num_words];
        let rem = row_len & 63;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last = (1u64 << rem) - 1;
            }
        }
        Self { words, row_len }
    }

    /// 从已有字切片构造位图，并自动裁剪超出 `row_len` 的尾部高位
    #[inline]
    pub fn from_words(words_slice: &[u64], row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        let mut words = smallvec::SmallVec::<[u64; 4]>::with_capacity(num_words);
        for i in 0..num_words {
            words.push(words_slice.get(i).copied().unwrap_or(0));
        }
        let rem = row_len & 63;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
        Self { words, row_len }
    }

    #[inline(always)]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline(always)]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize) {
        if row < self.row_len {
            self.words[row >> 6] |= 1u64 << (row & 63);
        }
    }

    #[inline(always)]
    pub fn clear_bit(&mut self, row: usize) {
        if row < self.row_len {
            self.words[row >> 6] &= !(1u64 << (row & 63));
        }
    }

    #[inline(always)]
    pub fn test(&self, row: usize) -> bool {
        if row >= self.row_len {
            return false;
        }
        (self.words[row >> 6] & (1u64 << (row & 63))) != 0
    }

    #[inline]
    pub fn is_all_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 与另一位图或字切片按位求交（`self &= other`）
    #[inline]
    pub fn intersect_words(&mut self, other_words: &[u64]) {
        for (i, w) in self.words.iter_mut().enumerate() {
            *w &= other_words.get(i).copied().unwrap_or(0);
        }
    }

    /// 与另一位图按位求交（`self &= other`）
    #[inline]
    pub fn intersect_with(&mut self, other: &RowBitMask) {
        self.intersect_words(&other.words);
    }

    /// 与另一位图或字切片按位求差（`self &= !other`）
    #[inline]
    pub fn difference_words(&mut self, other_words: &[u64]) {
        let rem = self.row_len & 63;
        let num_words = self.words.len();
        for (i, w) in self.words.iter_mut().enumerate() {
            let ow = other_words.get(i).copied().unwrap_or(0);
            *w &= !ow;
            if i + 1 == num_words && rem != 0 {
                *w &= (1u64 << rem) - 1;
            }
        }
    }

    /// 利用硬件 `trailing_zeros` 指令跳跃遍历所有置 1 的行索引
    #[inline]
    pub fn for_each_set_row<F: FnMut(usize)>(&self, mut f: F) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            let base = word_idx << 6;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let row = base + bit;
                if row < self.row_len {
                    f(row);
                }
                w &= w - 1;
            }
        }
    }
}

/// 关系对目标实体的表内二级倒排位图缓存（按 `columns_version` 自动惰性同步）
#[derive(Default)]
pub(crate) struct PairTargetIndexCache {
    pub by_relation: BTreeMap<ComponentTypeId, (u64, BTreeMap<EntityId, RowBitMask>)>,
}

/// Archetype 结构体，持有具有相同组件集合的所有实体及其列数据
pub(crate) struct Archetype {
    id: ArchetypeId,
    component_types: BTreeSet<ComponentTypeId>,
    component_types_slice: Vec<ComponentTypeId>,
    entities: Vec<EntityId>,
    /// 与 `entities` 同步维护的行级代际槽位表（支持 swap_remove 尾部实体 O(1) 槽位反查；若外部直接修改 entities_mut 则自动清空回退）
    entity_slots: Vec<LocalSlotId>,
    columns: BTreeMap<ComponentTypeId, Box<dyn AnyColumn>>,
    /// 与 `component_types_slice` 1:1 对应的扁平列指针表
    flat_columns: smallvec::SmallVec<[FlatColumnEntry; 8]>,
    /// 标记列是否可能处于分段态、列长未对齐或扁平列指针缓存待重建
    may_be_segmented_or_unaligned: bool,
    /// 列容器结构与可变借用版本号（任何可能替换/移除/分段化或修改列数据的操作均递增）
    columns_version: u64,
    /// 单组件插入原型转移边缓存 (ComponentTypeId, Rust TypeId -> (Target ArchetypeId, Target Column Index))
    add_edges: smallvec::SmallVec<[(ComponentTypeId, std::any::TypeId, u32, u16); 4]>,
    /// 单组件移除原型转移边缓存 (ComponentTypeId, Rust TypeId -> (Target ArchetypeId, Source Column Index))
    remove_edges: smallvec::SmallVec<[(ComponentTypeId, std::any::TypeId, u32, u16); 4]>,
    /// 非迁移行级状态位平面（`flag_id -> Vec<u64>`，每 64 行 1 个 `u64`）
    flag_planes: BTreeMap<u8, smallvec::SmallVec<[u64; 4]>>,
    /// 与 `component_types_slice` 1:1 对齐的列级 64 行块变更纪元表（惰性展开：为空时全列块共享 `default_chunk_epoch`）
    col_chunk_epochs: smallvec::SmallVec<[smallvec::SmallVec<[u64; 4]>; 8]>,
    /// 尚未发生单行细粒度分化时的默认块变更纪元
    default_chunk_epoch: u64,
    /// 关系对 `Pair<R>` 目标实体二级位图索引缓存
    pair_target_index: std::sync::RwLock<PairTargetIndexCache>,
}

#[allow(dead_code)]
impl Archetype {
    pub fn new(id: ArchetypeId, component_types: BTreeSet<ComponentTypeId>) -> Self {
        let component_types_slice: Vec<ComponentTypeId> =
            component_types.iter().copied().collect();
        let col_count = component_types_slice.len();
        let may_be_segmented_or_unaligned = !component_types.is_empty();
        Self {
            id,
            component_types,
            component_types_slice,
            entities: Vec::new(),
            entity_slots: Vec::new(),
            columns: BTreeMap::new(),
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned,
            columns_version: 1,
            add_edges: smallvec::SmallVec::new(),
            remove_edges: smallvec::SmallVec::new(),
            flag_planes: BTreeMap::new(),
            col_chunk_epochs: smallvec::smallvec![smallvec::SmallVec::new(); col_count],
            default_chunk_epoch: 1,
            pair_target_index: std::sync::RwLock::new(PairTargetIndexCache::default()),
        }
    }

    pub fn empty_with_schema(&self) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|(&type_id, col)| (type_id, col.empty_clone()))
            .collect();
        let col_count = self.component_types_slice.len();
        let mut arch = Self {
            id: self.id,
            component_types: self.component_types.clone(),
            component_types_slice: self.component_types_slice.clone(),
            entities: Vec::new(),
            entity_slots: Vec::new(),
            columns,
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned: true,
            columns_version: 1,
            add_edges: self.add_edges.clone(),
            remove_edges: self.remove_edges.clone(),
            flag_planes: BTreeMap::new(),
            col_chunk_epochs: smallvec::smallvec![smallvec::SmallVec::new(); col_count],
            default_chunk_epoch: 1,
            pair_target_index: std::sync::RwLock::new(PairTargetIndexCache::default()),
        };
        arch.rebuild_flat_columns();
        arch
    }

    /// 重建扁平连续列指针表并核验列非分段与行长一致性
    pub(crate) fn rebuild_flat_columns(&mut self) {
        self.flat_columns.clear();
        if self.col_chunk_epochs.len() != self.component_types_slice.len() {
            self.col_chunk_epochs
                .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
        }
        let expected_len = self.entities.len();
        let mut dirty = self.columns.len() != self.component_types_slice.len();
        for (&type_id, col) in &mut self.columns {
            if col.is_segmented()
                || col.len() != expected_len
                || col.as_ref().type_id() != type_id
                || ComponentTypeId::from_name(col.type_name()) != type_id
                || col.rust_type_id().is_none()
            {
                dirty = true;
                break;
            }
            let (Some(move_row_fn), Some(swap_remove_fn), Some(push_erased_fn)) = (
                col.raw_move_row_direct_fn(),
                col.raw_swap_remove_direct_fn(),
                col.raw_push_erased_direct_fn(),
            ) else {
                dirty = true;
                break;
            };
            let rust_type = col.rust_type_id();
            let col_ptr: *mut dyn AnyColumn = col.as_mut();
            self.flat_columns.push(FlatColumnEntry {
                type_id,
                rust_type,
                col_ptr,
                move_row_fn,
                swap_remove_fn,
                push_erased_fn,
            });
        }
        if !dirty
            && !self
                .flat_columns
                .iter()
                .map(|e| e.type_id)
                .eq(self.component_types_slice.iter().copied())
        {
            dirty = true;
        }
        if dirty {
            self.flat_columns.clear();
        }
        self.may_be_segmented_or_unaligned = dirty;
    }

    #[inline(always)]
    pub(crate) fn may_be_segmented_or_unaligned(&self) -> bool {
        self.may_be_segmented_or_unaligned
    }

    #[inline(always)]
    pub(crate) fn flat_columns(&self) -> &[FlatColumnEntry] {
        &self.flat_columns
    }

    #[inline(always)]
    pub(crate) fn add_edge(
        &self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
    ) -> Option<(u32, u16)> {
        self.add_edges
            .iter()
            .find_map(|&(cid, rtid, target, col_idx)| {
                (cid == type_id && rtid == rust_type).then_some((target, col_idx))
            })
    }

    #[inline]
    pub(crate) fn set_add_edge(
        &mut self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
        target: u32,
        dst_col_idx: u16,
    ) {
        if let Some((_, existing_rtid, existing_target, existing_col)) =
            self.add_edges.iter_mut().find(|(cid, _, _, _)| *cid == type_id)
        {
            *existing_rtid = rust_type;
            *existing_target = target;
            *existing_col = dst_col_idx;
        } else {
            self.add_edges
                .push((type_id, rust_type, target, dst_col_idx));
        }
    }

    #[inline(always)]
    pub(crate) fn remove_edge(
        &self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
    ) -> Option<(u32, u16)> {
        self.remove_edges
            .iter()
            .find_map(|&(cid, rtid, target, col_idx)| {
                (cid == type_id && rtid == rust_type).then_some((target, col_idx))
            })
    }

    #[inline]
    pub(crate) fn set_remove_edge(
        &mut self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
        target: u32,
        src_col_idx: u16,
    ) {
        if let Some((_, existing_rtid, existing_target, existing_col)) =
            self.remove_edges.iter_mut().find(|(cid, _, _, _)| *cid == type_id)
        {
            *existing_rtid = rust_type;
            *existing_target = target;
            *existing_col = src_col_idx;
        } else {
            self.remove_edges
                .push((type_id, rust_type, target, src_col_idx));
        }
    }

    #[inline(always)]
    pub(crate) fn has_segmented_columns(&self) -> bool {
        if !self.may_be_segmented_or_unaligned {
            return false;
        }
        self.columns.values().any(|col| col.is_segmented())
    }

    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    #[inline]
    pub fn component_types(&self) -> &BTreeSet<ComponentTypeId> {
        &self.component_types
    }

    #[inline]
    pub fn component_types_slice(&self) -> &[ComponentTypeId] {
        &self.component_types_slice
    }

    #[inline]
    pub fn entities(&self) -> &[EntityId] {
        &self.entities
    }

    #[inline(always)]
    pub(crate) fn entity_slot(&self, row: usize) -> Option<LocalSlotId> {
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.get(row).copied()
        } else {
            None
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    #[inline]
    pub(crate) fn columns(&self) -> &BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> {
        &self.columns
    }

    #[inline]
    pub(crate) fn columns_version(&self) -> u64 {
        self.columns_version
    }

    #[inline]
    pub(crate) fn columns_mut(&mut self) -> &mut BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> {
        self.columns_version = self.columns_version.wrapping_add(1);
        self.may_be_segmented_or_unaligned = true;
        self.flat_columns.clear();
        self.add_edges.clear();
        self.remove_edges.clear();
        &mut self.columns
    }

    /// 获取强类型列裸指针（供静态多元组插入缓存校验使用，不改变列集合结构）
    #[inline]
    pub(crate) fn raw_typed_column_ptr<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Option<*mut TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        let col = self.columns.get_mut(&type_id)?;
        if col.type_name() != std::any::type_name::<T>() || col.is_segmented() {
            return None;
        }
        let typed = col.as_any_mut().downcast_mut::<TypedColumn<T>>()?;
        Some(typed as *mut TypedColumn<T>)
    }

    /// 确保当前 Archetype 拥有源 Archetype 的所有列结构 (克隆空列)
    pub fn ensure_columns_from(&mut self, src: &Archetype) {
        self.columns_version = self.columns_version.wrapping_add(1);
        for (&type_id, col) in &src.columns {
            self.columns
                .entry(type_id)
                .or_insert_with(|| col.empty_clone());
            self.component_types.insert(type_id);
        }
        self.component_types_slice = self.component_types.iter().copied().collect();
        self.col_chunk_epochs
            .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
        self.add_edges.clear();
        self.remove_edges.clear();
        self.rebuild_flat_columns();
    }

    /// 注册一列强类型组件存储
    pub fn register_column<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        type_name: &'static str,
    ) {
        self.columns_version = self.columns_version.wrapping_add(1);
        let type_id = ComponentTypeId::from_name(type_name);
        self.component_types.insert(type_id);
        self.component_types_slice = self.component_types.iter().copied().collect();
        self.col_chunk_epochs
            .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
        self.columns
            .entry(type_id)
            .or_insert_with(|| Box::new(TypedColumn::<T>::new(type_name)));
        self.add_edges.clear();
        self.remove_edges.clear();
        self.rebuild_flat_columns();
    }

    /// 清空原型内所有实体与组件行数据，保留底层列容量与原型转移边缓存
    #[inline]
    pub(crate) fn clear_rows(&mut self) {
        self.columns_version = self.columns_version.wrapping_add(1);
        self.entities.clear();
        self.entity_slots.clear();
        self.flag_planes.clear();
        for chunks in &mut self.col_chunk_epochs {
            chunks.clear();
        }
        for col in self.columns.values_mut() {
            col.clear_rows();
        }
        if let Ok(mut cache) = self.pair_target_index.write() {
            cache.by_relation.clear();
        }
        self.rebuild_flat_columns();
    }

    #[inline]
    pub(crate) fn entities_mut(&mut self) -> &mut Vec<EntityId> {
        self.columns_version = self.columns_version.wrapping_add(1);
        self.entity_slots.clear();
        self.may_be_segmented_or_unaligned = true;
        &mut self.entities
    }

    /// 安全拆分实体列表切片与组件列可变映射，供连续可变查询使用
    #[inline]
    pub(crate) fn split_for_query_mut(
        &mut self,
    ) -> (
        &[EntityId],
        &[ComponentTypeId],
        &[FlatColumnEntry],
        bool,
        &mut BTreeMap<ComponentTypeId, Box<dyn AnyColumn>>,
    ) {
        self.columns_version = self.columns_version.wrapping_add(1);
        (
            &self.entities,
            &self.component_types_slice,
            &self.flat_columns,
            self.may_be_segmented_or_unaligned,
            &mut self.columns,
        )
    }

    #[inline]
    pub(crate) fn bump_columns_version_in_place(&mut self) {
        self.columns_version = self.columns_version.wrapping_add(1);
    }

    /// 在 `entities.push` 后同步维护非迁移位平面容量与已展开的块纪元向量
    #[inline(always)]
    fn on_push_row(&mut self, row: usize) {
        self.columns_version = self.columns_version.wrapping_add(1);
        let word_idx = row >> 6;
        let bit_mask = 1u64 << (row & 63);
        if !self.flag_planes.is_empty() {
            for words in self.flag_planes.values_mut() {
                if word_idx >= words.len() {
                    words.resize(word_idx + 1, 0);
                } else {
                    words[word_idx] &= !bit_mask;
                }
            }
        }
        let epoch = self.default_chunk_epoch;
        for chunks in &mut self.col_chunk_epochs {
            if !chunks.is_empty() {
                if word_idx >= chunks.len() {
                    chunks.resize(word_idx + 1, epoch);
                } else {
                    chunks[word_idx] = chunks[word_idx].max(epoch);
                }
            }
        }
    }

    /// 在 `swap_remove(row)` 时同步把 `last_row` 的状态位搬移至 `row`，并维护块变更纪元
    #[inline(always)]
    fn on_swap_remove_row(&mut self, row: usize, last_row: usize) {
        self.columns_version = self.columns_version.wrapping_add(1);
        let new_len = last_row;
        let new_words = (new_len + 63) >> 6;
        if !self.flag_planes.is_empty() {
            for words in self.flag_planes.values_mut() {
                let last_val = if (last_row >> 6) < words.len() {
                    (words[last_row >> 6] & (1u64 << (last_row & 63))) != 0
                } else {
                    false
                };
                if row < last_row {
                    let row_word = row >> 6;
                    if row_word >= words.len() && last_val {
                        words.resize(row_word + 1, 0);
                    }
                    if row_word < words.len() {
                        if last_val {
                            words[row_word] |= 1u64 << (row & 63);
                        } else {
                            words[row_word] &= !(1u64 << (row & 63));
                        }
                    }
                }
                if (last_row >> 6) < words.len() {
                    words[last_row >> 6] &= !(1u64 << (last_row & 63));
                }
                if words.len() > new_words {
                    words.truncate(new_words);
                }
            }
        }
        let row_chunk = row >> 6;
        let last_chunk = last_row >> 6;
        for chunks in &mut self.col_chunk_epochs {
            if !chunks.is_empty() {
                if row < last_row && row_chunk < chunks.len() {
                    let last_ep = chunks
                        .get(last_chunk)
                        .copied()
                        .unwrap_or(self.default_chunk_epoch);
                    chunks[row_chunk] = chunks[row_chunk].max(last_ep).max(self.default_chunk_epoch);
                }
                if chunks.len() > new_words {
                    chunks.truncate(new_words);
                }
            }
        }
    }

    /// 设置指定行的非迁移状态位（0..=255），返回修改前该位的旧布尔值
    #[inline]
    pub(crate) fn set_row_flag(&mut self, row: usize, flag: u8, value: bool) -> bool {
        if row >= self.entities.len() {
            return false;
        }
        let word_idx = row >> 6;
        let mask = 1u64 << (row & 63);
        let num_words = (self.entities.len() + 63) >> 6;
        if !value {
            if let Some(words) = self.flag_planes.get_mut(&flag) {
                if word_idx < words.len() {
                    let old = (words[word_idx] & mask) != 0;
                    words[word_idx] &= !mask;
                    return old;
                }
            }
            return false;
        }
        let words = self
            .flag_planes
            .entry(flag)
            .or_insert_with(|| smallvec::smallvec![0u64; num_words]);
        if words.len() <= word_idx {
            words.resize(num_words.max(word_idx + 1), 0);
        }
        let old = (words[word_idx] & mask) != 0;
        words[word_idx] |= mask;
        old
    }

    /// 读取指定行的非迁移状态位（0..=255）
    #[inline]
    pub(crate) fn get_row_flag(&self, row: usize, flag: u8) -> bool {
        if row >= self.entities.len() {
            return false;
        }
        let Some(words) = self.flag_planes.get(&flag) else {
            return false;
        };
        let word_idx = row >> 6;
        if word_idx >= words.len() {
            return false;
        }
        (words[word_idx] & (1u64 << (row & 63))) != 0
    }

    /// 获取指定状态位平面的底层 `u64` 字切片
    #[inline]
    pub(crate) fn flag_plane_words(&self, flag: u8) -> Option<&[u64]> {
        self.flag_planes.get(&flag).map(|v| v.as_slice())
    }

    /// 在实体跨 Archetype 迁移时，将源 `src_row` 的所有状态位同步复制到目标 `dst` 的 `dst_row`
    #[inline]
    pub(crate) fn copy_row_flags_to(&self, src_row: usize, dst: &mut Archetype, dst_row: usize) {
        if self.flag_planes.is_empty() {
            return;
        }
        let src_word = src_row >> 6;
        let src_mask = 1u64 << (src_row & 63);
        for (&flag, words) in &self.flag_planes {
            if src_word < words.len() && (words[src_word] & src_mask) != 0 {
                dst.set_row_flag(dst_row, flag, true);
            }
        }
    }

    /// 在 `reorder_entities` 重排实体顺序后，按 `old_rows_in_new_order` 同步重排行级状态位并刷新块纪元
    pub(crate) fn reorder_row_metadata(&mut self, old_rows_in_new_order: &[usize], epoch: u64) {
        self.columns_version = self.columns_version.wrapping_add(1);
        let new_len = old_rows_in_new_order.len();
        let num_words = (new_len + 63) >> 6;
        if !self.flag_planes.is_empty() {
            for words in self.flag_planes.values_mut() {
                let mut new_words = smallvec::smallvec![0u64; num_words];
                for (new_row, &old_row) in old_rows_in_new_order.iter().enumerate() {
                    let old_w = old_row >> 6;
                    if old_w < words.len() && (words[old_w] & (1u64 << (old_row & 63))) != 0 {
                        new_words[new_row >> 6] |= 1u64 << (new_row & 63);
                    }
                }
                *words = new_words;
            }
        }
        self.default_chunk_epoch = self.default_chunk_epoch.max(epoch);
        for chunks in &mut self.col_chunk_epochs {
            chunks.clear();
        }
    }

    /// 标记指定列位置 `col_pos` 在 `row` 所在 64 行块的变更纪元
    #[inline]
    pub(crate) fn mark_col_pos_row_epoch(&mut self, col_pos: usize, row: usize, epoch: u64) {
        if col_pos >= self.col_chunk_epochs.len() {
            self.col_chunk_epochs
                .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
            if col_pos >= self.col_chunk_epochs.len() {
                return;
            }
        }
        let num_chunks = (self.entities.len().max(row + 1) + 63) >> 6;
        let chunk_idx = row >> 6;
        let def_ep = self.default_chunk_epoch;
        let chunks = &mut self.col_chunk_epochs[col_pos];
        if chunks.is_empty() {
            chunks.resize(num_chunks, def_ep);
        } else if chunks.len() <= chunk_idx {
            chunks.resize(num_chunks.max(chunk_idx + 1), def_ep);
        }
        chunks[chunk_idx] = epoch;
    }

    /// 标记指定 `ComponentTypeId` 在 `row` 所在 64 行块的变更纪元
    #[inline]
    pub(crate) fn mark_column_row_epoch(
        &mut self,
        cid: ComponentTypeId,
        row: usize,
        epoch: u64,
    ) {
        if let Some(pos) = self.component_types_slice.iter().position(|&c| c == cid) {
            self.mark_col_pos_row_epoch(pos, row, epoch);
        }
    }

    /// 标记指定 `ComponentTypeId` 全列所有 64 行块的变更纪元
    #[inline]
    pub(crate) fn mark_column_all_chunks_epoch(&mut self, cid: ComponentTypeId, epoch: u64) {
        if let Some(pos) = self.component_types_slice.iter().position(|&c| c == cid) {
            if pos >= self.col_chunk_epochs.len() {
                self.col_chunk_epochs
                    .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
            }
            if pos < self.col_chunk_epochs.len() {
                let num_chunks = (self.entities.len() + 63) >> 6;
                let chunks = &mut self.col_chunk_epochs[pos];
                chunks.clear();
                chunks.resize(num_chunks, epoch);
            }
        }
    }

    /// 仅标记指定 `ComponentTypeId` 在位图掩码非零的 64 行块上的变更纪元
    #[inline]
    pub(crate) fn mark_column_masked_chunks_epoch(
        &mut self,
        cid: ComponentTypeId,
        mask: &RowBitMask,
        epoch: u64,
    ) {
        if let Some(pos) = self.component_types_slice.iter().position(|&c| c == cid) {
            if pos >= self.col_chunk_epochs.len() {
                self.col_chunk_epochs
                    .resize(self.component_types_slice.len(), smallvec::SmallVec::new());
            }
            if pos < self.col_chunk_epochs.len() {
                let num_chunks = (self.entities.len() + 63) >> 6;
                let def_ep = self.default_chunk_epoch;
                let chunks = &mut self.col_chunk_epochs[pos];
                if chunks.len() < num_chunks {
                    chunks.resize(num_chunks, def_ep);
                }
                for (w_idx, &word) in mask.words().iter().enumerate() {
                    if word != 0 && w_idx < chunks.len() {
                        chunks[w_idx] = epoch;
                    }
                }
            }
        }
    }

    /// 标记某一行所有组件列所在 64 行块的变更纪元（例如新实体插入或跨表迁入该行）
    #[inline(always)]
    pub(crate) fn mark_all_columns_row_epoch(&mut self, row: usize, epoch: u64) {
        self.default_chunk_epoch = self.default_chunk_epoch.max(epoch);
        let chunk_idx = row >> 6;
        let num_chunks = (self.entities.len().max(row + 1) + 63) >> 6;
        let def_ep = self.default_chunk_epoch;
        for chunks in &mut self.col_chunk_epochs {
            if !chunks.is_empty() {
                if chunks.len() <= chunk_idx {
                    chunks.resize(num_chunks.max(chunk_idx + 1), def_ep);
                }
                chunks[chunk_idx] = epoch;
            }
        }
    }

    /// 获取指定列的 64 行块变更纪元切片与默认块纪元
    #[inline]
    pub(crate) fn column_chunk_epochs(&self, cid: ComponentTypeId) -> Option<(&[u64], u64)> {
        let pos = self.component_types_slice.iter().position(|&c| c == cid)?;
        let slice = self
            .col_chunk_epochs
            .get(pos)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        Some((slice, self.default_chunk_epoch))
    }

    /// 查询或惰性构建关系对 `Pair<R>` 指向 `target` 实体的行级位图掩码
    pub(crate) fn pair_target_row_mask<
        R: ModelComponent + Serialize + DeserializeOwned + 'static,
    >(
        &self,
        target: EntityId,
    ) -> Option<RowBitMask> {
        let cid = ComponentTypeId::of::<crate::al::Pair<R>>();
        let ver = self.columns_version;
        let row_len = self.entities.len();
        if let Ok(guard) = self.pair_target_index.read() {
            if let Some((cached_ver, target_map)) = guard.by_relation.get(&cid) {
                if *cached_ver == ver {
                    return Some(
                        target_map
                            .get(&target)
                            .cloned()
                            .unwrap_or_else(|| RowBitMask::new_empty(row_len)),
                    );
                }
            }
        }
        let col = self.get_column::<crate::al::Pair<R>>()?;
        let mut target_map: BTreeMap<EntityId, RowBitMask> = BTreeMap::new();
        if let Some(slice) = col.as_slice() {
            for (row, pair) in slice.iter().enumerate() {
                if row < row_len {
                    target_map
                        .entry(pair.target)
                        .or_insert_with(|| RowBitMask::new_empty(row_len))
                        .set(row);
                }
            }
        } else {
            for (row, pair) in col.iter().enumerate() {
                if row < row_len {
                    target_map
                        .entry(pair.target)
                        .or_insert_with(|| RowBitMask::new_empty(row_len))
                        .set(row);
                }
            }
        }
        let result = target_map
            .get(&target)
            .cloned()
            .unwrap_or_else(|| RowBitMask::new_empty(row_len));
        if let Ok(mut guard) = self.pair_target_index.write() {
            guard.by_relation.insert(cid, (ver, target_map));
        }
        Some(result)
    }

    /// 向当前 Archetype 添加一个实体
    #[inline]
    pub fn push_entity(&mut self, entity: EntityId) -> usize {
        self.entity_slots.clear();
        let row = self.entities.len();
        self.entities.push(entity);
        self.on_push_row(row);
        row
    }

    /// 向当前 Archetype 同步添加实体及其代际槽位
    #[inline]
    pub(crate) fn push_entity_with_slot(&mut self, entity: EntityId, slot: LocalSlotId) -> usize {
        let row = self.entities.len();
        if self.entity_slots.len() == row {
            self.entity_slots.push(slot);
        } else {
            self.entity_slots.clear();
        }
        self.entities.push(entity);
        self.on_push_row(row);
        row
    }

    /// 仅对实体与行槽位执行同步 swap_remove（由已完成全列同步迁移的受控路径调用）
    #[inline]
    pub(crate) fn swap_remove_entity_only(&mut self, row: usize) {
        let last_row = self.entities.len() - 1;
        self.on_swap_remove_row(row, last_row);
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.swap_remove(row);
        } else {
            self.entity_slots.clear();
        }
        self.entities.swap_remove(row);
    }

    /// 获取强类型组件列不可变引用
    #[inline]
    pub fn get_column<T: ModelComponent + Serialize + DeserializeOwned>(
        &self,
    ) -> Option<&TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == type_id)?;
            let entry = unsafe { self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let typed_ptr = entry.col_ptr as *const () as *const TypedColumn<T>;
                return Some(unsafe { &*typed_ptr });
            }
            return None;
        }
        self.columns
            .get(&type_id)
            .and_then(|col| col.as_any().downcast_ref::<TypedColumn<T>>())
    }

    /// 获取强类型组件列可变引用（保持 flat_columns 缓存有效，因为不改变列集合与 Box 地址）
    #[inline]
    pub fn get_column_mut<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
    ) -> Option<&mut TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == type_id)?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                self.columns_version = self.columns_version.wrapping_add(1);
                let typed_ptr = entry.col_ptr as *mut () as *mut TypedColumn<T>;
                return Some(unsafe { &mut *typed_ptr });
            }
        }
        self.columns_version = self.columns_version.wrapping_add(1);
        self.columns
            .get_mut(&type_id)
            .and_then(|c| c.as_any_mut().downcast_mut::<TypedColumn<T>>())
    }

    /// 按 ComponentTypeId 获取强类型组件列可变引用，并校验类型一致性
    #[inline]
    pub fn get_typed_column_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        cid: ComponentTypeId,
        entity: EntityId,
    ) -> Result<&mut TypedColumn<T>, StateError> {
        self.columns_version = self.columns_version.wrapping_add(1);
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let typed_ptr = entry.col_ptr as *mut () as *mut TypedColumn<T>;
                return Ok(unsafe { &mut *typed_ptr });
            }
        }
        let col =
            self.columns
                .get_mut(&cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
        let type_name = col.type_name();
        col.as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().to_string(),
                actual: type_name.to_string(),
            })
    }

    /// 按 ComponentTypeId 获取强类型组件列裸指针（用于多列互斥借用，调用方必须保证 cid 互不相同）
    #[inline]
    pub fn get_typed_column_mut_ptr<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        cid: ComponentTypeId,
        entity: EntityId,
    ) -> Result<*mut TypedColumn<T>, StateError> {
        self.columns_version = self.columns_version.wrapping_add(1);
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                return Ok(entry.col_ptr as *mut () as *mut TypedColumn<T>);
            }
        }
        let col =
            self.columns
                .get_mut(&cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
        let type_name = col.type_name();
        let typed = col
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().to_string(),
                actual: type_name.to_string(),
            })?;
        Ok(typed as *mut TypedColumn<T>)
    }

    /// 为实体列表及所有组件列预留容量
    pub fn reserve(&mut self, additional: usize) {
        self.entities.reserve(additional);
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.reserve(additional);
        }
        for col in self.columns.values_mut() {
            col.reserve(additional);
        }
    }

    /// Prepare every fallible Clone before publishing any replacement. Existing
    /// Direct columns allocate nothing; uniquely owned frozen blocks are moved.
    #[inline]
    pub fn co_materialize_segmented_columns(&mut self) -> Result<(), StateError> {
        let mut replacements =
            smallvec::SmallVec::<[(ComponentTypeId, Box<dyn AnyColumn>); 4]>::new();
        let mut move_unique = false;
        for (&cid, col) in &mut self.columns {
            if let Some(prepared) = col.prepare_contiguous()? {
                replacements.push((cid, prepared));
            } else {
                move_unique = true;
            }
        }
        if !replacements.is_empty() || move_unique {
            self.columns_version = self.columns_version.wrapping_add(1);
            for (cid, replacement) in &mut replacements {
                std::mem::swap(
                    self.columns.get_mut(cid).expect("prepared column exists"),
                    replacement,
                );
            }
            if move_unique {
                for col in self.columns.values_mut() {
                    col.ensure_contiguous()?;
                }
            }
            drop(replacements);
            self.rebuild_flat_columns();
        }
        Ok(())
    }

    /// Swap-Remove 移除指定行实体，并返回被移动到该位置填补空缺的实体（若发生移动）
    pub fn swap_remove(&mut self, row: usize) -> Result<Option<EntityId>, StateError> {
        if row >= self.entities.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {} out of bounds in archetype {}",
                row, self.id.0
            )));
        }
        self.co_materialize_segmented_columns()?;
        Ok(self.swap_remove_materialized(row))
    }

    /// 在前置校验保证行号有效的前提下，执行不可失败的 Swap-Remove
    pub fn swap_remove_infallible(&mut self, row: usize) -> Option<EntityId> {
        if self.may_be_segmented_or_unaligned {
            self.co_materialize_segmented_columns()
                .expect("validated structural column materialization");
        }
        self.swap_remove_materialized(row)
    }

    fn swap_remove_materialized(&mut self, row: usize) -> Option<EntityId> {
        let last_row = self.entities.len() - 1;
        let is_last = row == last_row;
        self.on_swap_remove_row(row, last_row);
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.swap_remove(row);
        } else {
            self.entity_slots.clear();
        }
        self.entities.swap_remove(row);

        if !self.may_be_segmented_or_unaligned && self.flat_columns.len() == self.columns.len() {
            for entry in &self.flat_columns {
                unsafe {
                    (entry.swap_remove_fn)(entry.col_ptr as *mut (), row);
                }
            }
        } else {
            for col in self.columns.values_mut() {
                col.swap_remove(row);
            }
        }

        if is_last {
            None
        } else {
            // 被移到当前 row 的实体
            Some(self.entities[row])
        }
    }
}

impl Clone for Archetype {
    fn clone(&self) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|(&type_id, col)| (type_id, col.clone_box()))
            .collect();
        let mut arch = Self {
            id: self.id,
            component_types: self.component_types.clone(),
            component_types_slice: self.component_types_slice.clone(),
            entities: self.entities.clone(),
            entity_slots: self.entity_slots.clone(),
            columns,
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned: true,
            columns_version: 1,
            add_edges: self.add_edges.clone(),
            remove_edges: self.remove_edges.clone(),
            flag_planes: self.flag_planes.clone(),
            col_chunk_epochs: self.col_chunk_epochs.clone(),
            default_chunk_epoch: self.default_chunk_epoch,
            pair_target_index: std::sync::RwLock::new(PairTargetIndexCache::default()),
        };
        arch.rebuild_flat_columns();
        arch
    }
}

impl std::fmt::Debug for Archetype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Archetype")
            .field("id", &self.id)
            .field("component_types", &self.component_types)
            .field("entities_count", &self.entities.len())
            .field("columns_count", &self.columns.len())
            .finish()
    }
}

/// 按 ArchetypeId 快速查找原型（优先按索引直接定位，失败后降级线性遍历）
#[inline]
pub(crate) fn find_archetype_by_id<'a>(
    archetypes: &'a [Archetype],
    id: ArchetypeId,
) -> Option<&'a Archetype> {
    archetypes
        .get(id.0 as usize)
        .filter(|a| a.id() == id)
        .or_else(|| archetypes.iter().find(|a| a.id() == id))
}

#[cfg(test)]
#[path = "archetype_tests.rs"]
mod materialization_tests;
