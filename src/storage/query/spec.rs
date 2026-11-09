//! Borrowed, allocation-free queries over archetype-local component columns.

use crate::storage::archetype::{Archetype, RowBitMask};
use crate::id::ArchetypeId;
#[allow(unused_imports)]
use crate::storage::column::{ColumnDescriptor, TypedColumn, ColumnIter};
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::id::ComponentTypeId;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;
use std::ops::Range;

/// 原型只读头元数据抽象，用于跨表形态统一匹配原型。


pub(crate) trait ArchetypeHeader {
    fn archetype_id(&self) -> ArchetypeId;
    fn contains_component(&self, cid: ComponentTypeId) -> bool;
    #[inline]
    fn component_column_index(&self, cid: ComponentTypeId) -> Option<u16> {
        if self.contains_component(cid) {
            Some(u16::MAX)
        } else {
            None
        }
    }
}

impl ArchetypeHeader for Archetype {
    #[inline]
    fn archetype_id(&self) -> ArchetypeId {
        self.id()
    }

    #[inline]
    fn contains_component(&self, cid: ComponentTypeId) -> bool {
        self.component_types_slice().contains(&cid)
    }

    #[inline]
    fn component_column_index(&self, cid: ComponentTypeId) -> Option<u16> {
        self.component_types_slice()
            .iter()
            .position(|&id| id == cid)
            .and_then(|idx| u16::try_from(idx).ok())
    }
}

/// A supported read-only query shape.
///
/// Supported shapes are `&T`, tuples of two to four shared component references,
/// and `(EntityId, Q)`. Every shape yields exactly one outer entity identity:
/// `&T` yields `(EntityId, &T)`, and `(&A, &B)` yields `(EntityId, (&A, &B))`.
/// The explicit `(EntityId, Q)` spelling is equivalent to `Q`, not a second ID.
///
/// This trait is sealed: physical tables, slice extraction and cursor mutation
/// remain internal. Mutable references are deliberately unsupported.
///
/// ```compile_fail
/// struct ExternalQuery;
/// impl<'a> sumeru::QueryData<'a> for ExternalQuery {
///     type Item = ();
/// }
/// ```
#[allow(private_bounds)]
pub trait QueryData<'a>: sealed::Fetch<'a> {
    type Item: 'a;
    type Chunk: 'a;
}

/// A supported mutable query shape.
///
/// Supported shapes are `&T`, `&mut T`, tuples of 1 to 16 component references (e.g. `(&mut A, &B)` or
/// `(&mut A, &mut B)`), and `(EntityId, Q)`.
///
/// This trait is sealed: physical tables, slice extraction and cursor mutation
/// remain internal. Shared references use the same shape machinery for shared
/// Store access. For ordinary read-only Core access, prefer `store.query::<&T>()`
/// to avoid the exclusive borrow and epoch advance of `query_mut`.
///
/// ```compile_fail
/// struct ExternalMutQuery;
/// impl<'a> sumeru::QueryMutData<'a> for ExternalMutQuery {
///     type Item = ();
///     type Chunk = ();
/// }
/// ```
#[allow(private_bounds)]
pub trait QueryMutData<'a>: sealed::FetchMut<'a> {
    type Item: 'a;
    type Chunk: 'a;
    /// Validates query shape and anti-aliasing requirements fail-closed.
    #[doc(hidden)]
    fn validate_query() {}
}

/// A parameter within a mutable query tuple (`&'a mut T` or `&'a T`).
#[allow(private_bounds)]
pub trait QueryMutParam<'a>: sealed::ParamFetch<'a> {
    type Item: 'a;
    type Chunk: 'a;
}

/// 行级列内存访问器抽象，杜绝行级过滤时反向构造整列切片破坏并发别名规则
pub(crate) trait RowColumnAccessor {
    fn get_column_descriptor(&self, cid: ComponentTypeId) -> Option<ColumnDescriptor>;
    fn len(&self) -> usize;

    #[inline]
    fn flag_plane_words(&self, flag: u8) -> Option<&[u64]> {
        let _ = flag;
        None
    }

    #[inline]
    fn row_has_flag(&self, row: usize, flag: u8) -> bool {
        let _ = (row, flag);
        false
    }

    #[inline]
    fn column_chunk_epochs(&self, cid: ComponentTypeId) -> Option<(&[u64], u64)> {
        let _ = cid;
        None
    }

    #[inline]
    fn pair_target_row_mask<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
        target: EntityId,
    ) -> Option<RowBitMask> {
        let _ = target;
        None
    }
}

impl RowColumnAccessor for Archetype {
    #[inline]
    fn get_column_descriptor(&self, cid: ComponentTypeId) -> Option<ColumnDescriptor> {
        if !self.may_be_segmented_or_unaligned()
            && self.flat_columns().len() == self.component_types_slice().len()
        {
            let pos = self.component_types_slice().iter().position(|&id| id == cid)?;
            let entry = unsafe { *self.flat_columns().get_unchecked(pos) };
            return Some(ColumnDescriptor::from_column_ref(unsafe { &*entry.col_ptr }));
        }
        self.columns().get(&cid).map(|c| ColumnDescriptor::from_column_ref(c.as_ref()))
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn flag_plane_words(&self, flag: u8) -> Option<&[u64]> {
        self.flag_plane_words(flag)
    }

    #[inline]
    fn row_has_flag(&self, row: usize, flag: u8) -> bool {
        self.get_row_flag(row, flag)
    }

    #[inline]
    fn column_chunk_epochs(&self, cid: ComponentTypeId) -> Option<(&[u64], u64)> {
        self.column_chunk_epochs(cid)
    }

    #[inline]
    fn pair_target_row_mask<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
        target: EntityId,
    ) -> Option<RowBitMask> {
        self.pair_target_row_mask::<R>(target)
    }
}

/// 供 `QueryFilter` 在块级（Archetype / Table）或行级判断组件存在性的探针接口。
pub trait ColumnChunkProbe {
    /// 当前块内是否整体包含组件 `T`。
    fn chunk_has_column<T: 'static>(&self) -> bool;

    /// 当前块内第 `row` 行是否包含组件 `T`（对于标准 Archetype 表恒等于 `chunk_has_column::<T>()`）。
    #[inline]
    fn row_has_component<T: 'static>(&self, _row: usize) -> bool {
        self.chunk_has_column::<T>()
    }
}

impl ColumnChunkProbe for Archetype {
    #[inline]
    fn chunk_has_column<T: 'static>(&self) -> bool {
        self.component_types_slice()
            .contains(&ComponentTypeId::of::<T>())
    }
}

/// Sealed table-level predicate implemented by the builder's filter chain.
#[allow(private_bounds)]
pub trait QueryFilter: sealed::Filter + Send + Sync + 'static {
    /// 块级预筛：若整个块必定不满足过滤条件则返回 `false`，可直接跳过整张表。
    #[inline]
    fn matches_chunk<P: ColumnChunkProbe>(probe: &P) -> bool {
        <Self as sealed::Filter>::matches_chunk_probe(probe)
    }

    /// 块级探针行复核：对于稀疏或混合表按行过滤，对于标准 Archetype 表恒为 `true`。
    #[inline]
    fn matches_row_probe<P: ColumnChunkProbe>(probe: &P, row: usize) -> bool {
        <Self as sealed::Filter>::matches_row_probe(probe, row)
    }

    /// Metadata-only conjunction, used to validate a staged candidate universe.
    /// Row predicates are proven separately by protected component read heads.
    #[doc(hidden)]
    fn archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
        <Self as sealed::Filter>::collect_archetype_requirements(self, out);
    }
    /// 编译期标记：当前过滤器链中是否存在行级过滤条件（用于零开销内联代码分发）
    const IS_ROW_FILTERED: bool = <Self as sealed::Filter>::IS_ROW_FILTERED_INTERNAL;

    /// 检查指定的原型表是否匹配当前过滤器（支持任何实现 ArchetypeHeader 的原型形态）
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn matches_archetype<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
        <Self as sealed::Filter>::matches_archetype_internal(self, archetype)
    }

    /// 检查指定的原型表及行索引是否满足过滤器（通过 RowColumnAccessor 抽象支持 Archetype 与 SharedArchetype，消除整列切片借用）
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn matches_row<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
        <Self as sealed::Filter>::matches_row_internal(self, accessor, row)
    }

    /// 检查当前过滤器链中是否存在行级过滤条件
    #[inline]
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn is_row_filtered(&self) -> bool {
        Self::IS_ROW_FILTERED
    }

    /// 批量收集匹配行位图掩码：基于 64-bit word 位图与分块代际实现 O(N/64) 极速过滤与跳步
    #[inline]
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn collect_matched_row_mask<A: RowColumnAccessor + ?Sized>(
        &self,
        accessor: &A,
    ) -> Option<RowBitMask> {
        if !Self::IS_ROW_FILTERED {
            return None;
        }
        let total = accessor.len();
        if total == 0 {
            return Some(RowBitMask::new_empty(0));
        }
        let mut mask = RowBitMask::new_all_set(total);
        if !<Self as sealed::Filter>::apply_row_mask(self, accessor, &mut mask) {
            return Some(RowBitMask::new_empty(total));
        }
        Some(mask)
    }

    /// 批量收集匹配行索引：整张表仅解析一次列描述符后连续判断
    #[inline]
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn collect_matched_rows<A: RowColumnAccessor + ?Sized>(&self, accessor: &A) -> Option<Vec<usize>> {
        if !Self::IS_ROW_FILTERED {
            return None;
        }
        let total = accessor.len();
        if total == 0 {
            return Some(Vec::new());
        }
        let Some(state) = self.prepare_table(accessor) else {
            return Some(Vec::new());
        };
        let mut rows = Vec::new();
        for r in 0..total {
            if self.matches_row_prepared(&state, r) {
                if rows.capacity() == 0 {
                    // 首次命中才分配堆内存；上限保守限制为 256（或 total），
                    // 小表零扩容，大表稀疏场景彻底杜绝巨额预分配
                    rows.reserve(total.min(256));
                }
                rows.push(r);
            }
        }
        Some(rows)
    }

    /// 遍历当前过滤器中涉及读取组件的类型描述符
    fn filter_read_components(&self, mut f: impl FnMut(ComponentTypeId, &'static str)) {
        let mut map = std::collections::BTreeMap::new();
        <Self as sealed::Filter>::collect_read_components(self, &mut map);
        for (cid, name) in map {
            f(cid, name);
        }
    }

    /// 返回当前过滤器链中涉及读取的 `RowFlags` 位掩码（`bit i == 1` 表示读取 `FLAG = i`）
    #[inline]
    fn filter_read_flags(&self) -> u8 {
        <Self as sealed::Filter>::collect_read_flags(self)
    }
}

// Borrow-local predicate preparation, shared by grouped selection and native
// filters. Caller holds READ protection for every filter component in `range`.
pub(crate) fn visit_matched_range<F: QueryFilter, A: RowColumnAccessor>(
    filter: &F, accessor: &A, range: impl IntoIterator<Item = usize>, mut visit: impl FnMut(usize),
) {
    if let Some(prepared) = filter.prepare_table(accessor) {
        for row in range {
            if filter.matches_row_prepared(&prepared, row) { visit(row); }
        }
    }
}

/// 强类型组件字段描述符，用于零开销提取特定不可变字段值进行硬边界比对。
pub trait ComponentField<C>: 'static + Send + Sync {
    type Value: 'static + PartialEq + Send + Sync + Clone + std::fmt::Debug;
    fn extract<'a>(component: &'a C) -> &'a Self::Value;
    fn field_name() -> &'static str;
}

/// 基于组件特定字段相等性匹配的硬边界过滤器节点。
pub struct FilterEq<F, C, Field: ComponentField<C>> {
    pub(crate) previous: F,
    pub(crate) expected: Field::Value,
    pub(crate) marker: PhantomData<(C, Field)>,
}

impl<F: Clone, C, Field: ComponentField<C>> Clone for FilterEq<F, C, Field> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            previous: self.previous.clone(),
            expected: self.expected.clone(),
            marker: PhantomData,
        }
    }
}

impl<F: std::fmt::Debug, C, Field: ComponentField<C>> std::fmt::Debug for FilterEq<F, C, Field> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterEq")
            .field("previous", &self.previous)
            .field("field", &Field::field_name())
            .field("expected", &self.expected)
            .finish()
    }
}

impl<F, C, Field: ComponentField<C>> FilterEq<F, C, Field> {
    #[inline]
    pub fn new(previous: F, expected: Field::Value) -> Self {
        Self {
            previous,
            expected,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn expected(&self) -> &Field::Value {
        &self.expected
    }
}

/// 基于受信任业务闭包谓词的过滤器节点。
#[derive(Clone)]
pub struct BusinessFilter<F, C> {
    pub(crate) previous: F,
    pub(crate) predicate: std::sync::Arc<dyn Fn(&C) -> bool + Send + Sync + 'static>,
    pub(crate) marker: PhantomData<C>,
}

impl<F, C> BusinessFilter<F, C> {
    #[inline]
    pub fn new(previous: F, predicate: impl Fn(&C) -> bool + Send + Sync + 'static) -> Self {
        Self {
            previous,
            predicate: std::sync::Arc::new(predicate),
            marker: PhantomData,
        }
    }
}

impl<F: std::fmt::Debug, C> std::fmt::Debug for BusinessFilter<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BusinessFilter")
            .field("previous", &self.previous)
            .field("component", &std::any::type_name::<C>())
            .finish()
    }
}


/// Initial query predicate, matching every table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct All;

/// Type-level chain node requiring component T, without reading its values.
#[derive(Clone, Copy, Debug)]
pub struct With<T, F = All> {
    pub(crate) previous: F,
    pub(crate) type_id: ComponentTypeId,
    pub(crate) marker: PhantomData<fn() -> T>,
}

impl<T: 'static, F> With<T, F> {
    #[inline]
    pub fn new(previous: F) -> Self {
        Self {
            previous,
            type_id: ComponentTypeId::of::<T>(),
            marker: PhantomData,
        }
    }
}

impl<T: 'static, F: Default> Default for With<T, F> {
    #[inline]
    fn default() -> Self {
        Self::new(F::default())
    }
}

/// Type-level chain node excluding component T, without reading its values.
#[derive(Clone, Copy, Debug)]
pub struct Without<T, F = All> {
    pub(crate) previous: F,
    pub(crate) type_id: ComponentTypeId,
    pub(crate) marker: PhantomData<fn() -> T>,
}

impl<T: 'static, F> Without<T, F> {
    #[inline]
    pub fn new(previous: F) -> Self {
        Self {
            previous,
            type_id: ComponentTypeId::of::<T>(),
            marker: PhantomData,
        }
    }
}

impl<T: 'static, F: Default> Default for Without<T, F> {
    #[inline]
    fn default() -> Self {
        Self::new(F::default())
    }
}

/// 非迁移位图状态标记包含过滤节点（O(1) 64-bit word 按位与）。
#[derive(Clone, Copy, Debug, Default)]
pub struct WithFlag<const FLAG: u8, F = All> {
    pub(crate) previous: F,
}

impl<const FLAG: u8, F> WithFlag<FLAG, F> {
    #[inline]
    pub const fn new(previous: F) -> Self {
        Self { previous }
    }
}

/// 非迁移位图状态标记排除过滤节点（O(1) 64-bit word 按位差）。
#[derive(Clone, Copy, Debug, Default)]
pub struct WithoutFlag<const FLAG: u8, F = All> {
    pub(crate) previous: F,
}

impl<const FLAG: u8, F> WithoutFlag<FLAG, F> {
    #[inline]
    pub const fn new(previous: F) -> Self {
        Self { previous }
    }
}

/// 基于 64 行分块写代际（Chunk Write Epoch）的增量脏数据过滤节点。
#[derive(Clone, Copy, Debug)]
pub struct ChangedSince<T, F = All> {
    pub(crate) previous: F,
    pub(crate) since_epoch: u64,
    pub(crate) marker: PhantomData<fn() -> T>,
}

impl<T: 'static, F> ChangedSince<T, F> {
    #[inline]
    pub const fn new(previous: F, since_epoch: u64) -> Self {
        Self {
            previous,
            since_epoch,
            marker: PhantomData,
        }
    }
}

/// 基于原型内嵌目标二级位图索引的关系对 `(R, target)` 过滤节点。
#[derive(Clone, Copy, Debug)]
pub struct WithPairTarget<R, F = All> {
    pub(crate) previous: F,
    pub(crate) target: EntityId,
    pub(crate) marker: PhantomData<fn() -> R>,
}

impl<R: 'static, F> WithPairTarget<R, F> {
    #[inline]
    pub const fn new(previous: F, target: EntityId) -> Self {
        Self {
            previous,
            target,
            marker: PhantomData,
        }
    }
}

impl<F: sealed::Filter + Send + Sync + 'static> QueryFilter for F {}

pub(crate) mod sealed {
    use super::*;

    pub(crate) trait Fetch<'a> {
        type State;
        #[inline]
        fn total_row_width() -> usize {
            0
        }
        fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)>;
        #[inline(always)]
        fn fast_prepare_bound(
            archetype: &'a Archetype,
            col_slots: &[u16],
            start: usize,
        ) -> Option<(Self::State, usize)> {
            let _ = col_slots;
            Self::prepare(archetype, start)
        }
        fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item
        where
            Self: QueryData<'a>;
        fn skip_rows(state: &mut Self::State, count: usize);
        fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)>
        where
            Self: QueryData<'a>;
        #[inline(always)]
        fn fast_chunk_bound(
            archetype: &'a Archetype,
            col_slots: &[u16],
            start: usize,
        ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)>
        where
            Self: QueryData<'a>,
        {
            let _ = col_slots;
            Self::chunk(archetype, start)
        }
    }

    pub(crate) trait FetchMut<'a> {
        type State;
        #[inline]
        fn total_row_width() -> usize {
            0
        }
        fn prepare(
            columns: &[(ComponentTypeId, ColumnDescriptor)],
            range: Range<usize>,
        ) -> Option<Self::State>;
        fn fast_prepare_archetype(
            archetype: &'a mut Archetype,
        ) -> Option<(Self::State, &'a [EntityId])>;
        #[inline(always)]
        fn fast_prepare_bound(
            archetype: &'a mut Archetype,
            col_slots: &[u16],
        ) -> Option<(Self::State, &'a [EntityId])> {
            let _ = col_slots;
            Self::fast_prepare_archetype(archetype)
        }
        fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item
        where
            Self: QueryMutData<'a>;
        fn skip_rows(state: &mut Self::State, count: usize);
        fn chunk(
            entities: &'a [EntityId],
            columns: &[(ComponentTypeId, ColumnDescriptor)],
            range: Range<usize>,
        ) -> Option<<Self as QueryMutData<'a>>::Chunk>
        where
            Self: QueryMutData<'a>;
        fn fast_chunk_archetype(
            archetype: &'a mut Archetype,
        ) -> Option<<Self as QueryMutData<'a>>::Chunk>
        where
            Self: QueryMutData<'a>;
        #[inline(always)]
        fn fast_chunk_bound(
            archetype: &'a mut Archetype,
            col_slots: &[u16],
        ) -> Option<<Self as QueryMutData<'a>>::Chunk>
        where
            Self: QueryMutData<'a>,
        {
            let _ = col_slots;
            Self::fast_chunk_archetype(archetype)
        }
        #[inline(always)]
        fn mark_written_columns(
            archetype: &mut Archetype,
            masked: Option<&RowBitMask>,
            epoch: u64,
        ) {
            if epoch == 0 {
                return;
            }
            for (cid, _, is_write) in Self::access_descriptors() {
                if is_write {
                    if let Some(mask) = masked {
                        archetype.mark_column_masked_chunks_epoch(cid, mask, epoch);
                    } else {
                        archetype.mark_column_all_chunks_epoch(cid, epoch);
                    }
                }
            }
        }
        #[allow(dead_code)]
        fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]>;
        #[inline]
        fn required_mask() -> u16 {
            u16::MAX
        }
    }

    pub(crate) trait ReadParam<'a>: QueryMutParam<'a> {
        type ReadState: 'a;
        #[inline]
        fn element_size() -> usize {
            0
        }
        fn prepare_read(archetype: &'a Archetype, start: usize) -> Option<(Self::ReadState, usize)>;
        #[inline(always)]
        fn fast_prepare_read_slot(
            archetype: &'a Archetype,
            slot: u16,
            start: usize,
        ) -> Option<(Self::ReadState, usize)> {
            let _ = slot;
            Self::prepare_read(archetype, start)
        }
        fn next_read(state: &mut Self::ReadState) -> <Self as QueryMutParam<'a>>::Item;
        fn skip_read(state: &mut Self::ReadState, count: usize);
        fn chunk_read(
            archetype: &'a Archetype,
            start: usize,
        ) -> Option<(<Self as QueryMutParam<'a>>::Chunk, usize)>;
        #[inline(always)]
        fn fast_chunk_read_slot(
            archetype: &'a Archetype,
            slot: u16,
            start: usize,
        ) -> Option<(<Self as QueryMutParam<'a>>::Chunk, usize)> {
            let _ = slot;
            Self::chunk_read(archetype, start)
        }
        fn truncate_chunk(
            chunk: <Self as QueryMutParam<'a>>::Chunk,
            len: usize,
        ) -> <Self as QueryMutParam<'a>>::Chunk;
    }

    pub(crate) trait ParamFetch<'a> {
        type State: 'a;
        const IS_REQUIRED: bool = true;
        #[inline]
        fn element_size() -> usize {
            0
        }
        fn component_type_id() -> ComponentTypeId;
        fn type_name() -> &'static str;
        #[allow(dead_code)]
        fn is_write() -> bool;
        unsafe fn prepare(
            desc: Option<&ColumnDescriptor>,
            range: Range<usize>,
        ) -> Option<Self::State>;
        unsafe fn fast_prepare(
            comp_slice: &[ComponentTypeId],
            flat_cols: &[crate::storage::archetype::FlatColumnEntry],
            len: usize,
        ) -> Option<Self::State>;
        unsafe fn fast_prepare_slot(
            comp_slice: &[ComponentTypeId],
            flat_cols: &[crate::storage::archetype::FlatColumnEntry],
            slot: u16,
            len: usize,
        ) -> Option<Self::State>;
        unsafe fn chunk(
            desc: Option<&ColumnDescriptor>,
            range: Range<usize>,
        ) -> Option<<Self as QueryMutParam<'a>>::Chunk>
        where
            Self: QueryMutParam<'a>;
        unsafe fn fast_chunk(
            comp_slice: &[ComponentTypeId],
            flat_cols: &[crate::storage::archetype::FlatColumnEntry],
            len: usize,
        ) -> Option<<Self as QueryMutParam<'a>>::Chunk>
        where
            Self: QueryMutParam<'a>;
        unsafe fn fast_chunk_slot(
            comp_slice: &[ComponentTypeId],
            flat_cols: &[crate::storage::archetype::FlatColumnEntry],
            slot: u16,
            len: usize,
        ) -> Option<<Self as QueryMutParam<'a>>::Chunk>
        where
            Self: QueryMutParam<'a>;
        fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item
        where
            Self: QueryMutParam<'a>;
        fn skip_items(state: &mut Self::State, count: usize);
    }

    pub(crate) trait QueryMutShapeInternal<B> {
        unsafe fn resolve_archetype_buffers(
            get_desc: &dyn Fn(ComponentTypeId) -> Option<ColumnDescriptor>,
            expected_len: usize,
        ) -> Option<B>;
    }

    // Unsafe physical extraction belongs to the sealed implementation, not the
    // public shape contract. External compile-fail examples can therefore supply
    // every public requirement without naming private column descriptors.
    /// Physical extraction only. Callers keep the descriptor's allocation alive
    /// and stationary, check `start <= end <= len`, and hold a compatible lease
    /// for the full lifetime of every returned reference. `next_item` requires
    /// an unexhausted iterator. No method here acquires or extends a lease.
    #[allow(private_interfaces)]
    pub trait QueryMutParamShapeInternal {
        type Param<'a>: QueryMutParam<'a>;
        type ColumnBuffer: Send + Clone;
        type IterState<'a>;
        fn segment_end_with_cursor(buffer: &Self::ColumnBuffer, row: usize, cursor: &mut usize) -> usize;
        unsafe fn resolve_buffer(
            desc: &ColumnDescriptor,
            expected_len: usize,
        ) -> Option<Self::ColumnBuffer>;
        #[inline]
        unsafe fn resolve_buffer_from_desc(
            desc: Option<&ColumnDescriptor>,
            expected_len: usize,
        ) -> Option<Self::ColumnBuffer> {
            let desc = desc?;
            unsafe { Self::resolve_buffer(desc, expected_len) }
        }
        unsafe fn prepare_state<'view>(
            buffer: &Self::ColumnBuffer,
            range: Range<usize>,
            cursor: &mut usize,
        ) -> Self::IterState<'view>;
        unsafe fn next_item<'view>(
            state: &mut Self::IterState<'view>,
        ) -> <Self::Param<'view> as QueryMutParam<'view>>::Item;
        unsafe fn skip_items<'view>(
            state: &mut Self::IterState<'view>,
            count: usize,
        );
        unsafe fn chunk_state<'view>(
            buffer: &Self::ColumnBuffer,
            range: Range<usize>,
            cursor: &mut usize,
        ) -> <Self::Param<'view> as QueryMutParam<'view>>::Chunk;
    }

    use super::{ArchetypeHeader, RowColumnAccessor};

    pub(crate) trait Filter {
        const IS_ROW_FILTERED_INTERNAL: bool;
        type TableState<'a>;

        #[inline]
        fn matches_chunk_probe<P: super::ColumnChunkProbe>(_probe: &P) -> bool {
            true
        }

        #[inline]
        fn matches_row_probe<P: super::ColumnChunkProbe>(_probe: &P, _row: usize) -> bool {
            true
        }

        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>>;
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool;

        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool;
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>);
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>);

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            0
        }

        #[inline]
        fn update_since_epoch(&mut self, _epoch: u64) {}

        #[inline]
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            if let Some(state) = self.prepare_table(accessor) {
                self.matches_row_prepared(&state, row)
            } else {
                false
            }
        }

        /// 将当前过滤器的行级结果与 `mask` 进行原地按位与求交；若交集全为 0 则返回 `false` 立即短路。
        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(
            &self,
            accessor: &A,
            mask: &mut RowBitMask,
        ) -> bool {
            if !Self::IS_ROW_FILTERED_INTERNAL {
                return !mask.is_all_zero();
            }
            let Some(state) = self.prepare_table(accessor) else {
                mask.words_mut().fill(0);
                return false;
            };
            let row_len = mask.row_len();
            let mut any_non_zero = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let row = w_idx * 64 + bit;
                    if row < row_len && !self.matches_row_prepared(&state, row) {
                        *word &= !(1u64 << bit);
                    }
                    w &= w - 1;
                }
                if *word != 0 {
                    any_non_zero = true;
                }
            }
            any_non_zero
        }
    }

    impl Filter for All {
        fn collect_archetype_requirements(&self, _: &mut Vec<(ComponentTypeId, bool)>) {}
        const IS_ROW_FILTERED_INTERNAL: bool = false;
        type TableState<'a> = ();

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, _: &'a A) -> Option<Self::TableState<'a>> {
            Some(())
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, _: &Self::TableState<'a>, _: usize) -> bool {
            true
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, _: &H) -> bool {
            true
        }

        #[inline]
        fn collect_read_components(&self, _out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {}

        #[inline]
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, _: &A, _: usize) -> bool {
            true
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, _: &A, mask: &mut RowBitMask) -> bool {
            !mask.is_all_zero()
        }
    }

    impl<T: 'static, F: Filter> Filter for With<T, F> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((self.type_id, true));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = F::IS_ROW_FILTERED_INTERNAL;
        type TableState<'a> = F::TableState<'a>;

        #[inline]
        fn matches_chunk_probe<P: super::ColumnChunkProbe>(probe: &P) -> bool {
            F::matches_chunk_probe(probe) && probe.chunk_has_column::<T>()
        }

        #[inline]
        fn matches_row_probe<P: super::ColumnChunkProbe>(probe: &P, row: usize) -> bool {
            F::matches_row_probe(probe, row) && probe.row_has_component::<T>(row)
        }

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            self.previous.prepare_table(accessor)
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            self.previous.matches_row_prepared(state, row)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype) && archetype.contains_component(self.type_id)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            self.previous.matches_row_internal(accessor, row)
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            self.previous.apply_row_mask(accessor, mask)
        }
    }

    impl<T: 'static, F: Filter> Filter for Without<T, F> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((self.type_id, false));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = F::IS_ROW_FILTERED_INTERNAL;
        type TableState<'a> = F::TableState<'a>;

        #[inline]
        fn matches_chunk_probe<P: super::ColumnChunkProbe>(probe: &P) -> bool {
            F::matches_chunk_probe(probe) && !probe.chunk_has_column::<T>()
        }

        #[inline]
        fn matches_row_probe<P: super::ColumnChunkProbe>(probe: &P, row: usize) -> bool {
            F::matches_row_probe(probe, row) && !probe.row_has_component::<T>(row)
        }

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            self.previous.prepare_table(accessor)
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            self.previous.matches_row_prepared(state, row)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype) && !archetype.contains_component(self.type_id)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            self.previous.matches_row_internal(accessor, row)
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            self.previous.apply_row_mask(accessor, mask)
        }
    }

    impl<F1: Filter, F2: Filter> Filter for (F1, F2) {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.0.collect_archetype_requirements(out);
            self.1.collect_archetype_requirements(out);
        }
        const IS_ROW_FILTERED_INTERNAL: bool =
            F1::IS_ROW_FILTERED_INTERNAL || F2::IS_ROW_FILTERED_INTERNAL;
        type TableState<'a> = (F1::TableState<'a>, F2::TableState<'a>);

        #[inline]
        fn matches_chunk_probe<P: super::ColumnChunkProbe>(probe: &P) -> bool {
            F1::matches_chunk_probe(probe) && F2::matches_chunk_probe(probe)
        }

        #[inline]
        fn matches_row_probe<P: super::ColumnChunkProbe>(probe: &P, row: usize) -> bool {
            F1::matches_row_probe(probe, row) && F2::matches_row_probe(probe, row)
        }

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            Some((self.0.prepare_table(accessor)?, self.1.prepare_table(accessor)?))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            self.0.matches_row_prepared(&state.0, row) && self.1.matches_row_prepared(&state.1, row)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.0.matches_archetype_internal(archetype) && self.1.matches_archetype_internal(archetype)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.0.collect_read_components(out);
            self.1.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.0.collect_read_flags() | self.1.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.0.update_since_epoch(epoch);
            self.1.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            self.0.apply_row_mask(accessor, mask) && self.1.apply_row_mask(accessor, mask)
        }
    }

    impl<F1: Filter, F2: Filter, F3: Filter> Filter for (F1, F2, F3) {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.0.collect_archetype_requirements(out);
            self.1.collect_archetype_requirements(out);
            self.2.collect_archetype_requirements(out);
        }
        const IS_ROW_FILTERED_INTERNAL: bool =
            F1::IS_ROW_FILTERED_INTERNAL || F2::IS_ROW_FILTERED_INTERNAL || F3::IS_ROW_FILTERED_INTERNAL;
        type TableState<'a> = (F1::TableState<'a>, F2::TableState<'a>, F3::TableState<'a>);

        #[inline]
        fn matches_chunk_probe<P: super::ColumnChunkProbe>(probe: &P) -> bool {
            F1::matches_chunk_probe(probe) && F2::matches_chunk_probe(probe) && F3::matches_chunk_probe(probe)
        }

        #[inline]
        fn matches_row_probe<P: super::ColumnChunkProbe>(probe: &P, row: usize) -> bool {
            F1::matches_row_probe(probe, row) && F2::matches_row_probe(probe, row) && F3::matches_row_probe(probe, row)
        }

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            Some((
                self.0.prepare_table(accessor)?,
                self.1.prepare_table(accessor)?,
                self.2.prepare_table(accessor)?,
            ))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            self.0.matches_row_prepared(&state.0, row)
                && self.1.matches_row_prepared(&state.1, row)
                && self.2.matches_row_prepared(&state.2, row)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.0.matches_archetype_internal(archetype)
                && self.1.matches_archetype_internal(archetype)
                && self.2.matches_archetype_internal(archetype)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.0.collect_read_components(out);
            self.1.collect_read_components(out);
            self.2.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.0.collect_read_flags() | self.1.collect_read_flags() | self.2.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.0.update_since_epoch(epoch);
            self.1.update_since_epoch(epoch);
            self.2.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            self.0.apply_row_mask(accessor, mask)
                && self.1.apply_row_mask(accessor, mask)
                && self.2.apply_row_mask(accessor, mask)
        }
    }

    impl<F: Filter, C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>> Filter
        for FilterEq<F, C, Field>
    {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((ComponentTypeId::of::<C>(), true));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, ColumnDescriptor);

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev_state = self.previous.prepare_table(accessor)?;
            let desc = accessor.get_column_descriptor(ComponentTypeId::of::<C>())?;
            Some((prev_state, desc))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            if !self.previous.matches_row_prepared(&state.0, row) {
                return false;
            }
            let desc = &state.1;
            if row >= desc.len {
                return false;
            }
            // SAFETY:
            // 1. 调用方在验证行过滤前必须持有对该 row 的有效租约或独占 store 借用；
            // 2. row < desc.len；
            // 3. C 由 ComponentTypeId::of::<C>() 校验与列类型一致；
            // 4. get_unchecked 仅派生当前单行元素的引用，杜绝构造整列切片导致其他并发行的借用失效。
            let comp: &C = unsafe { desc.get_unchecked(row) };
            Field::extract(comp) == &self.expected
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
                && archetype.contains_component(ComponentTypeId::of::<C>())
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
            out.insert(ComponentTypeId::of::<C>(), std::any::type_name::<C>());
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            let Some(desc) = accessor.get_column_descriptor(ComponentTypeId::of::<C>()) else {
                mask.words_mut().fill(0);
                return false;
            };
            let row_len = mask.row_len().min(desc.len);
            let mut any_non_zero = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let row = w_idx * 64 + bit;
                    if row >= row_len {
                        *word &= !(1u64 << bit);
                    } else {
                        let comp: &C = unsafe { desc.get_unchecked(row) };
                        if Field::extract(comp) != &self.expected {
                            *word &= !(1u64 << bit);
                        }
                    }
                    w &= w - 1;
                }
                if *word != 0 {
                    any_non_zero = true;
                }
            }
            any_non_zero
        }
    }

    impl<F: Filter, C: ModelComponent + Serialize + DeserializeOwned> Filter for BusinessFilter<F, C> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((ComponentTypeId::of::<C>(), true));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, ColumnDescriptor);

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev_state = self.previous.prepare_table(accessor)?;
            let desc = accessor.get_column_descriptor(ComponentTypeId::of::<C>())?;
            Some((prev_state, desc))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            if !self.previous.matches_row_prepared(&state.0, row) {
                return false;
            }
            let desc = &state.1;
            if row >= desc.len {
                return false;
            }
            // SAFETY:
            // 1. 调用方在验证行过滤前必须持有对该 row 的有效租约或独占 store 借用；
            // 2. row < desc.len；
            // 3. C 由 ComponentTypeId::of::<C>() 校验与列类型一致；
            // 4. get_unchecked 仅派生当前单行元素的引用，杜绝构造整列切片导致其他并发行的借用失效。
            let comp: &C = unsafe { desc.get_unchecked(row) };
            (self.predicate)(comp)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
                && archetype.contains_component(ComponentTypeId::of::<C>())
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
            out.insert(ComponentTypeId::of::<C>(), std::any::type_name::<C>());
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            let Some(desc) = accessor.get_column_descriptor(ComponentTypeId::of::<C>()) else {
                mask.words_mut().fill(0);
                return false;
            };
            let row_len = mask.row_len().min(desc.len);
            let mut any_non_zero = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let row = w_idx * 64 + bit;
                    if row >= row_len {
                        *word &= !(1u64 << bit);
                    } else {
                        let comp: &C = unsafe { desc.get_unchecked(row) };
                        if !(self.predicate)(comp) {
                            *word &= !(1u64 << bit);
                        }
                    }
                    w &= w - 1;
                }
                if *word != 0 {
                    any_non_zero = true;
                }
            }
            any_non_zero
        }
    }

    impl<const FLAG: u8, F: Filter> Filter for WithFlag<FLAG, F> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, Option<&'a [u64]>);

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev = self.previous.prepare_table(accessor)?;
            let plane = accessor.flag_plane_words(FLAG);
            Some((prev, plane))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            if !self.previous.matches_row_prepared(&state.0, row) {
                return false;
            }
            let Some(words) = state.1 else {
                return false;
            };
            let w_idx = row >> 6;
            if w_idx >= words.len() {
                return false;
            }
            (words[w_idx] & (1u64 << (row & 63))) != 0
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags() | (1u8 << FLAG)
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            let Some(plane_words) = accessor.flag_plane_words(FLAG) else {
                mask.words_mut().fill(0);
                return false;
            };
            mask.intersect_words(plane_words);
            !mask.is_all_zero()
        }
    }

    impl<const FLAG: u8, F: Filter> Filter for WithoutFlag<FLAG, F> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, Option<&'a [u64]>);

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev = self.previous.prepare_table(accessor)?;
            let plane = accessor.flag_plane_words(FLAG);
            Some((prev, plane))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            if !self.previous.matches_row_prepared(&state.0, row) {
                return false;
            }
            let Some(words) = state.1 else {
                return true;
            };
            let w_idx = row >> 6;
            if w_idx >= words.len() {
                return true;
            }
            (words[w_idx] & (1u64 << (row & 63))) == 0
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags() | (1u8 << FLAG)
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            if let Some(plane_words) = accessor.flag_plane_words(FLAG) {
                mask.difference_words(plane_words);
            }
            !mask.is_all_zero()
        }
    }

    impl<T: 'static, F: Filter> Filter for ChangedSince<T, F> {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((ComponentTypeId::of::<T>(), true));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, (&'a [u64], u64));

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev = self.previous.prepare_table(accessor)?;
            let epochs = accessor.column_chunk_epochs(ComponentTypeId::of::<T>())?;
            Some((prev, epochs))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            if !self.previous.matches_row_prepared(&state.0, row) {
                return false;
            }
            let chunk_idx = row >> 6;
            let (epochs, def_ep) = state.1;
            let ep = epochs.get(chunk_idx).copied().unwrap_or(def_ep);
            ep > self.since_epoch
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
                && archetype.contains_component(ComponentTypeId::of::<T>())
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
            out.insert(ComponentTypeId::of::<T>(), std::any::type_name::<T>());
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
            self.since_epoch = epoch;
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            let Some((epochs, def_ep)) = accessor.column_chunk_epochs(ComponentTypeId::of::<T>()) else {
                mask.words_mut().fill(0);
                return false;
            };
            let mut any_non_zero = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                if *word == 0 {
                    continue;
                }
                let ep = epochs.get(w_idx).copied().unwrap_or(def_ep);
                if ep <= self.since_epoch {
                    *word = 0;
                } else {
                    any_non_zero = true;
                }
            }
            any_non_zero
        }
    }

    impl<R: ModelComponent + Serialize + DeserializeOwned + 'static, F: Filter> Filter
        for WithPairTarget<R, F>
    {
        fn collect_archetype_requirements(&self, out: &mut Vec<(ComponentTypeId, bool)>) {
            self.previous.collect_archetype_requirements(out);
            out.push((ComponentTypeId::of::<crate::al::Pair<R>>(), true));
        }
        const IS_ROW_FILTERED_INTERNAL: bool = true;
        type TableState<'a> = (F::TableState<'a>, RowBitMask);

        #[inline]
        fn prepare_table<'a, A: RowColumnAccessor + ?Sized>(&self, accessor: &'a A) -> Option<Self::TableState<'a>> {
            let prev = self.previous.prepare_table(accessor)?;
            let mask = if let Some(m) = accessor.pair_target_row_mask::<R>(self.target) {
                m
            } else {
                let desc = accessor.get_column_descriptor(ComponentTypeId::of::<crate::al::Pair<R>>())?;
                let mut m = RowBitMask::new_empty(desc.len);
                for r in 0..desc.len {
                    let pair: &crate::al::Pair<R> = unsafe { desc.get_unchecked(r) };
                    if pair.target == self.target {
                        m.set(r);
                    }
                }
                m
            };
            if mask.is_all_zero() {
                return None;
            }
            Some((prev, mask))
        }

        #[inline]
        fn matches_row_prepared<'a>(&self, state: &Self::TableState<'a>, row: usize) -> bool {
            self.previous.matches_row_prepared(&state.0, row) && state.1.test(row)
        }

        #[inline]
        fn matches_archetype_internal<H: ArchetypeHeader + ?Sized>(&self, archetype: &H) -> bool {
            self.previous.matches_archetype_internal(archetype)
                && archetype.contains_component(ComponentTypeId::of::<crate::al::Pair<R>>())
        }

        #[inline]
        fn collect_read_components(&self, out: &mut std::collections::BTreeMap<ComponentTypeId, &'static str>) {
            self.previous.collect_read_components(out);
            out.insert(
                ComponentTypeId::of::<crate::al::Pair<R>>(),
                std::any::type_name::<crate::al::Pair<R>>(),
            );
        }

        #[inline]
        fn collect_read_flags(&self) -> u8 {
            self.previous.collect_read_flags()
        }

        #[inline]
        fn update_since_epoch(&mut self, epoch: u64) {
            self.previous.update_since_epoch(epoch);
        }

        #[inline]
        fn apply_row_mask<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, mask: &mut RowBitMask) -> bool {
            if !self.previous.apply_row_mask(accessor, mask) {
                return false;
            }
            if let Some(pair_mask) = accessor.pair_target_row_mask::<R>(self.target) {
                mask.intersect_with(&pair_mask);
                return !mask.is_all_zero();
            }
            let Some(desc) = accessor.get_column_descriptor(ComponentTypeId::of::<crate::al::Pair<R>>()) else {
                mask.words_mut().fill(0);
                return false;
            };
            let row_len = mask.row_len().min(desc.len);
            let mut any_non_zero = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let row = w_idx * 64 + bit;
                    if row >= row_len {
                        *word &= !(1u64 << bit);
                    } else {
                        let pair: &crate::al::Pair<R> = unsafe { desc.get_unchecked(row) };
                        if pair.target != self.target {
                            *word &= !(1u64 << bit);
                        }
                    }
                    w &= w - 1;
                }
                if *word != 0 {
                    any_non_zero = true;
                }
            }
            any_non_zero
        }
    }
}

#[inline]
pub(crate) fn column_suffix<T: ModelComponent + Serialize + DeserializeOwned>(
    archetype: &Archetype,
    start: usize,
) -> Option<&[T]> {
    if start == 0
        && !archetype.may_be_segmented_or_unaligned()
        && archetype.flat_columns().len() == archetype.component_types_slice().len()
    {
        let cid = ComponentTypeId::of::<T>();
        let pos = archetype
            .component_types_slice()
            .iter()
            .position(|&id| id == cid)?;
        let entry = unsafe { *archetype.flat_columns().get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
        let slice = col.as_slice();
        if slice.len() != archetype.len() {
            return None;
        }
        return Some(slice);
    }
    let column = archetype.get_column::<T>()?;
    // Never zip/truncate an inconsistent table silently.
    if column.len() != archetype.len() { return None; }
    if start == 0 {
        if let Some(slice) = column.backing().as_direct_slice() {
            return Some(slice);
        }
    }
    column.backing().slice_from(start)
}

#[inline(always)]
pub(crate) fn column_suffix_at_slot<T: ModelComponent + Serialize + DeserializeOwned>(
    archetype: &Archetype,
    slot: u16,
    start: usize,
) -> Option<&[T]> {
    let slot_idx = slot as usize;
    if start == 0
        && !archetype.may_be_segmented_or_unaligned()
        && slot_idx < archetype.flat_columns().len()
        && archetype.flat_columns().len() == archetype.component_types_slice().len()
    {
        let entry = unsafe { *archetype.flat_columns().get_unchecked(slot_idx) };
        if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
            let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
            let slice = col.as_slice();
            if slice.len() == archetype.len() {
                return Some(slice);
            }
        }
    }
    column_suffix::<T>(archetype, start)
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ReadParam<'a> for &'a T {
    type ReadState = ColumnIter<'a, T>;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare_read(archetype: &'a Archetype, start: usize) -> Option<(Self::ReadState, usize)> {
        let col = archetype.get_column::<T>()?;
        if col.len() != archetype.len() {
            return None;
        }
        Some((col.iter_from(start), archetype.len()))
    }

    #[inline(always)]
    fn fast_prepare_read_slot(
        archetype: &'a Archetype,
        slot: u16,
        start: usize,
    ) -> Option<(Self::ReadState, usize)> {
        let slot_idx = slot as usize;
        if !archetype.may_be_segmented_or_unaligned()
            && slot_idx < archetype.flat_columns().len()
            && archetype.flat_columns().len() == archetype.component_types_slice().len()
        {
            let entry = unsafe { *archetype.flat_columns().get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                if col.len() == archetype.len() {
                    return Some((col.iter_from(start), archetype.len()));
                }
            }
        }
        Self::prepare_read(archetype, start)
    }

    #[inline]
    fn next_read(state: &mut Self::ReadState) -> &'a T {
        state
            .next()
            .expect("validated equal-length component and entity slices")
    }

    #[inline]
    fn skip_read(state: &mut Self::ReadState, count: usize) {
        state.skip_rows(count);
    }

    #[inline]
    fn chunk_read(archetype: &'a Archetype, start: usize) -> Option<(&'a [T], usize)> {
        let slice = column_suffix::<T>(archetype, start)?;
        Some((slice, slice.len()))
    }

    #[inline(always)]
    fn fast_chunk_read_slot(
        archetype: &'a Archetype,
        slot: u16,
        start: usize,
    ) -> Option<(&'a [T], usize)> {
        let slice = column_suffix_at_slot::<T>(archetype, slot, start)?;
        Some((slice, slice.len()))
    }

    #[inline]
    fn truncate_chunk(chunk: &'a [T], len: usize) -> &'a [T] {
        &chunk[..len]
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ReadParam<'a> for Option<&'a T> {
    type ReadState = Option<ColumnIter<'a, T>>;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare_read(archetype: &'a Archetype, start: usize) -> Option<(Self::ReadState, usize)> {
        let cid = ComponentTypeId::of::<T>();
        if !archetype.component_types_slice().contains(&cid) {
            return Some((None, archetype.len()));
        }
        let col = archetype.get_column::<T>()?;
        if col.len() != archetype.len() {
            return None;
        }
        Some((Some(col.iter_from(start)), archetype.len()))
    }

    #[inline(always)]
    fn fast_prepare_read_slot(
        archetype: &'a Archetype,
        slot: u16,
        start: usize,
    ) -> Option<(Self::ReadState, usize)> {
        if slot == u16::MAX {
            return Some((None, archetype.len()));
        }
        let slot_idx = slot as usize;
        if !archetype.may_be_segmented_or_unaligned()
            && slot_idx < archetype.flat_columns().len()
            && archetype.flat_columns().len() == archetype.component_types_slice().len()
        {
            let entry = unsafe { *archetype.flat_columns().get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                if col.len() == archetype.len() {
                    return Some((Some(col.iter_from(start)), archetype.len()));
                }
            }
        }
        Self::prepare_read(archetype, start)
    }

    #[inline]
    fn next_read(state: &mut Self::ReadState) -> Option<&'a T> {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component and entity slices"))
    }

    #[inline]
    fn skip_read(state: &mut Self::ReadState, count: usize) {
        if let Some(it) = state.as_mut() {
            it.skip_rows(count);
        }
    }

    #[inline]
    fn chunk_read(archetype: &'a Archetype, start: usize) -> Option<(Option<&'a [T]>, usize)> {
        let cid = ComponentTypeId::of::<T>();
        if !archetype.component_types_slice().contains(&cid) {
            return Some((None, archetype.len().saturating_sub(start)));
        }
        let slice = column_suffix::<T>(archetype, start)?;
        Some((Some(slice), slice.len()))
    }

    #[inline(always)]
    fn fast_chunk_read_slot(
        archetype: &'a Archetype,
        slot: u16,
        start: usize,
    ) -> Option<(Option<&'a [T]>, usize)> {
        if slot == u16::MAX {
            return Some((None, archetype.len().saturating_sub(start)));
        }
        let slice = column_suffix_at_slot::<T>(archetype, slot, start)?;
        Some((Some(slice), slice.len()))
    }

    #[inline]
    fn truncate_chunk(chunk: Option<&'a [T]>, len: usize) -> Option<&'a [T]> {
        chunk.map(|s| &s[..len])
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryData<'a> for &'a T {
    type Item = (EntityId, &'a T);
    type Chunk = (&'a [EntityId], &'a [T]);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::Fetch<'a> for &'a T {
    type State = ColumnIter<'a, T>;
    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }
    #[inline]
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        <&'a T as sealed::ReadParam<'a>>::prepare_read(archetype, start)
    }
    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(Self::State, usize)> {
        if let Some(&slot) = col_slots.first() {
            return <&'a T as sealed::ReadParam<'a>>::fast_prepare_read_slot(archetype, slot, start);
        }
        Self::prepare(archetype, start)
    }
    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        (entity, <&'a T as sealed::ReadParam<'a>>::next_read(state))
    }
    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <&'a T as sealed::ReadParam<'a>>::skip_read(state, count);
    }
    #[inline]
    fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        let slice = column_suffix::<T>(archetype, start)?;
        let end = start + slice.len();
        Some(((&archetype.entities()[start..end], slice), end))
    }
    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        if let Some(&slot) = col_slots.first() {
            let slice = column_suffix_at_slot::<T>(archetype, slot, start)?;
            let end = start + slice.len();
            return Some(((&archetype.entities()[start..end], slice), end));
        }
        <Self as sealed::Fetch<'a>>::chunk(archetype, start)
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryData<'a> for Option<&'a T> {
    type Item = (EntityId, Option<&'a T>);
    type Chunk = (&'a [EntityId], Option<&'a [T]>);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::Fetch<'a> for Option<&'a T> {
    type State = Option<ColumnIter<'a, T>>;
    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }
    #[inline]
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        <Option<&'a T> as sealed::ReadParam<'a>>::prepare_read(archetype, start)
    }
    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(Self::State, usize)> {
        if let Some(&slot) = col_slots.first() {
            return <Option<&'a T> as sealed::ReadParam<'a>>::fast_prepare_read_slot(archetype, slot, start);
        }
        Self::prepare(archetype, start)
    }
    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        (entity, <Option<&'a T> as sealed::ReadParam<'a>>::next_read(state))
    }
    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <Option<&'a T> as sealed::ReadParam<'a>>::skip_read(state, count);
    }
    #[inline]
    fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        let (opt_slice, avail) = <Option<&'a T> as sealed::ReadParam<'a>>::chunk_read(archetype, start)?;
        let end = start + avail;
        Some(((&archetype.entities()[start..end], opt_slice), end))
    }
    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        if let Some(&slot) = col_slots.first() {
            let (opt_slice, avail) = <Option<&'a T> as sealed::ReadParam<'a>>::fast_chunk_read_slot(archetype, slot, start)?;
            let end = start + avail;
            return Some(((&archetype.entities()[start..end], opt_slice), end));
        }
        <Self as sealed::Fetch<'a>>::chunk(archetype, start)
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutParam<'a> for &'a mut T {
    type Item = &'a mut T;
    type Chunk = &'a mut [T];
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ParamFetch<'a> for &'a mut T {
    type State = std::slice::IterMut<'a, T>;
    const IS_REQUIRED: bool = true;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn component_type_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }

    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }

    #[inline]
    fn is_write() -> bool {
        true
    }

    #[inline]
    unsafe fn prepare(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<Self::State> {
        let desc = desc?;
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) { return None; }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        // Descriptor was captured under exclusive Store access before the first
        // range is borrowed. Forward-only ranges never overlap.
        Some(unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(offset), range.len()).iter_mut() })
    }

    #[inline(always)]
    unsafe fn fast_prepare(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let pos = comp_slice.iter().position(|&id| id == cid)?;
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
        let slice = col.as_mut_slice();
        if slice.len() != len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len).iter_mut() })
    }

    #[inline(always)]
    unsafe fn fast_prepare_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<Self::State> {
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
                let slice = col.as_mut_slice();
                if slice.len() == len {
                    return Some(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len).iter_mut() });
                }
            }
        }
        unsafe { Self::fast_prepare(comp_slice, flat_cols, len) }
    }

    #[inline]
    unsafe fn chunk(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let desc = desc?;
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) { return None; }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(offset), range.len()) })
    }

    #[inline(always)]
    unsafe fn fast_chunk(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let pos = comp_slice.iter().position(|&id| id == cid)?;
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
        let slice = col.as_mut_slice();
        if slice.len() != len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len) })
    }

    #[inline(always)]
    unsafe fn fast_chunk_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
                let slice = col.as_mut_slice();
                if slice.len() == len {
                    return Some(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len) });
                }
            }
        }
        unsafe { Self::fast_chunk(comp_slice, flat_cols, len) }
    }

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .next()
            .expect("validated equal-length component slice")
    }

    #[inline]
    fn skip_items(state: &mut Self::State, count: usize) {
        if count > 0 {
            let _ = state.nth(count - 1);
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutParam<'a> for &'a T {
    type Item = &'a T;
    type Chunk = &'a [T];
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ParamFetch<'a> for &'a T {
    type State = std::slice::Iter<'a, T>;
    const IS_REQUIRED: bool = true;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn component_type_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }

    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }

    #[inline]
    fn is_write() -> bool {
        false
    }

    #[inline]
    unsafe fn prepare(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<Self::State> {
        let desc = desc?;
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) { return None; }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        // Descriptor was captured under exclusive Store access before the first
        // range is borrowed. Forward-only ranges never overlap.
        Some(unsafe { std::slice::from_raw_parts((base as *const T).add(offset), range.len()).iter() })
    }

    #[inline(always)]
    unsafe fn fast_prepare(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let pos = comp_slice.iter().position(|&id| id == cid)?;
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
        let slice = col.as_slice();
        if slice.len() != len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr(), len).iter() })
    }

    #[inline(always)]
    unsafe fn fast_prepare_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<Self::State> {
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                let slice = col.as_slice();
                if slice.len() == len {
                    return Some(unsafe { std::slice::from_raw_parts(slice.as_ptr(), len).iter() });
                }
            }
        }
        unsafe { Self::fast_prepare(comp_slice, flat_cols, len) }
    }

    #[inline]
    unsafe fn chunk(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let desc = desc?;
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) { return None; }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(unsafe { std::slice::from_raw_parts((base as *const T).add(offset), range.len()) })
    }

    #[inline(always)]
    unsafe fn fast_chunk(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let pos = comp_slice.iter().position(|&id| id == cid)?;
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
        let slice = col.as_slice();
        if slice.len() != len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr(), len) })
    }

    #[inline(always)]
    unsafe fn fast_chunk_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                let slice = col.as_slice();
                if slice.len() == len {
                    return Some(unsafe { std::slice::from_raw_parts(slice.as_ptr(), len) });
                }
            }
        }
        unsafe { Self::fast_chunk(comp_slice, flat_cols, len) }
    }

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .next()
            .expect("validated equal-length component slice")
    }

    #[inline]
    fn skip_items(state: &mut Self::State, count: usize) {
        if count > 0 {
            let _ = state.nth(count - 1);
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutParam<'a> for Option<&'a T> {
    type Item = Option<&'a T>;
    type Chunk = Option<&'a [T]>;
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ParamFetch<'a> for Option<&'a T> {
    type State = Option<std::slice::Iter<'a, T>>;
    const IS_REQUIRED: bool = false;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn component_type_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }

    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }

    #[inline]
    fn is_write() -> bool {
        false
    }

    #[inline]
    unsafe fn prepare(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<Self::State> {
        let Some(desc) = desc else {
            return Some(None);
        };
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(Some(unsafe {
            std::slice::from_raw_parts((base as *const T).add(offset), range.len()).iter()
        }))
    }

    #[inline(always)]
    unsafe fn fast_prepare(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let Some(pos) = comp_slice.iter().position(|&id| id == cid) else {
            return Some(None);
        };
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
        let slice = col.as_slice();
        if slice.len() != len {
            return None;
        }
        Some(Some(unsafe {
            std::slice::from_raw_parts(slice.as_ptr(), len).iter()
        }))
    }

    #[inline(always)]
    unsafe fn fast_prepare_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<Self::State> {
        if slot == u16::MAX {
            return Some(None);
        }
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                let slice = col.as_slice();
                if slice.len() == len {
                    return Some(Some(unsafe {
                        std::slice::from_raw_parts(slice.as_ptr(), len).iter()
                    }));
                }
            }
        }
        unsafe { Self::fast_prepare(comp_slice, flat_cols, len) }
    }

    #[inline]
    unsafe fn chunk(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let Some(desc) = desc else {
            return Some(None);
        };
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(Some(unsafe {
            std::slice::from_raw_parts((base as *const T).add(offset), range.len())
        }))
    }

    #[inline(always)]
    unsafe fn fast_chunk(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let Some(pos) = comp_slice.iter().position(|&id| id == cid) else {
            return Some(None);
        };
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
        let slice = col.as_slice();
        if slice.len() != len {
            return None;
        }
        Some(Some(unsafe {
            std::slice::from_raw_parts(slice.as_ptr(), len)
        }))
    }

    #[inline(always)]
    unsafe fn fast_chunk_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        if slot == u16::MAX {
            return Some(None);
        }
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
                let slice = col.as_slice();
                if slice.len() == len {
                    return Some(Some(unsafe {
                        std::slice::from_raw_parts(slice.as_ptr(), len)
                    }));
                }
            }
        }
        unsafe { Self::fast_chunk(comp_slice, flat_cols, len) }
    }

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component slice"))
    }

    #[inline]
    fn skip_items(state: &mut Self::State, count: usize) {
        if count > 0 {
            if let Some(it) = state.as_mut() {
                let _ = it.nth(count - 1);
            }
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutParam<'a> for Option<&'a mut T> {
    type Item = Option<&'a mut T>;
    type Chunk = Option<&'a mut [T]>;
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ParamFetch<'a>
    for Option<&'a mut T>
{
    type State = Option<std::slice::IterMut<'a, T>>;
    const IS_REQUIRED: bool = false;

    #[inline]
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn component_type_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }

    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }

    #[inline]
    fn is_write() -> bool {
        true
    }

    #[inline]
    unsafe fn prepare(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<Self::State> {
        let Some(desc) = desc else {
            return Some(None);
        };
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(Some(unsafe {
            std::slice::from_raw_parts_mut((base as *mut T).add(offset), range.len()).iter_mut()
        }))
    }

    #[inline(always)]
    unsafe fn fast_prepare(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let Some(pos) = comp_slice.iter().position(|&id| id == cid) else {
            return Some(None);
        };
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
        let slice = col.as_mut_slice();
        if slice.len() != len {
            return None;
        }
        Some(Some(unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len).iter_mut()
        }))
    }

    #[inline(always)]
    unsafe fn fast_prepare_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<Self::State> {
        if slot == u16::MAX {
            return Some(None);
        }
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
                let slice = col.as_mut_slice();
                if slice.len() == len {
                    return Some(Some(unsafe {
                        std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len).iter_mut()
                    }));
                }
            }
        }
        unsafe { Self::fast_prepare(comp_slice, flat_cols, len) }
    }

    #[inline]
    unsafe fn chunk(
        desc: Option<&ColumnDescriptor>,
        range: Range<usize>,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let Some(desc) = desc else {
            return Some(None);
        };
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let (base, offset, end) = desc.locate(range.start);
        assert!(range.end <= end);
        Some(Some(unsafe {
            std::slice::from_raw_parts_mut((base as *mut T).add(offset), range.len())
        }))
    }

    #[inline(always)]
    unsafe fn fast_chunk(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let Some(pos) = comp_slice.iter().position(|&id| id == cid) else {
            return Some(None);
        };
        let entry = unsafe { *flat_cols.get_unchecked(pos) };
        if entry.rust_type != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
        let slice = col.as_mut_slice();
        if slice.len() != len {
            return None;
        }
        Some(Some(unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len)
        }))
    }

    #[inline(always)]
    unsafe fn fast_chunk_slot(
        comp_slice: &[ComponentTypeId],
        flat_cols: &[crate::storage::archetype::FlatColumnEntry],
        slot: u16,
        len: usize,
    ) -> Option<<Self as QueryMutParam<'a>>::Chunk> {
        if slot == u16::MAX {
            return Some(None);
        }
        let slot_idx = slot as usize;
        if slot_idx < flat_cols.len() {
            let entry = unsafe { *flat_cols.get_unchecked(slot_idx) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let col = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
                let slice = col.as_mut_slice();
                if slice.len() == len {
                    return Some(Some(unsafe {
                        std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len)
                    }));
                }
            }
        }
        unsafe { Self::fast_chunk(comp_slice, flat_cols, len) }
    }

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component slice"))
    }

    #[inline]
    fn skip_items(state: &mut Self::State, count: usize) {
        if count > 0 {
            if let Some(it) = state.as_mut() {
                let _ = it.nth(count - 1);
            }
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutData<'a> for Option<&'a mut T> {
    type Item = (EntityId, Option<&'a mut T>);
    type Chunk = (&'a [EntityId], Option<&'a mut [T]>);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::FetchMut<'a> for Option<&'a mut T> {
    type State = Option<std::slice::IterMut<'a, T>>;

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare(
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        unsafe { <Self as sealed::ParamFetch<'a>>::prepare(desc, range) }
    }

    #[inline(always)]
    fn fast_prepare_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let state = unsafe { <Self as sealed::ParamFetch<'a>>::fast_prepare(comp_slice, flat_cols, len)? };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        if state.is_some() {
            archetype.bump_columns_version_in_place();
        }
        Some((state, entities))
    }

    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let state = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_prepare_slot(
                comp_slice,
                flat_cols,
                col_slots.first().copied().unwrap_or(u16::MAX),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        if state.is_some() {
            archetype.bump_columns_version_in_place();
        }
        Some((state, entities))
    }

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        (entity, <Self as sealed::ParamFetch<'a>>::next_item(state))
    }

    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <Self as sealed::ParamFetch<'a>>::skip_items(state, count);
    }

    #[inline]
    fn chunk(
        entities: &'a [EntityId],
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        let c = unsafe { <Self as sealed::ParamFetch<'a>>::chunk(desc, range)? };
        Some((entities, c))
    }

    #[inline(always)]
    fn fast_chunk_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let chunk = unsafe { <Self as sealed::ParamFetch<'a>>::fast_chunk(comp_slice, flat_cols, len)? };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        if chunk.is_some() {
            archetype.bump_columns_version_in_place();
        }
        Some((entities, chunk))
    }

    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let chunk = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_chunk_slot(
                comp_slice,
                flat_cols,
                col_slots.first().copied().unwrap_or(u16::MAX),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        if chunk.is_some() {
            archetype.bump_columns_version_in_place();
        }
        Some((entities, chunk))
    }

    #[inline]
    fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
        smallvec::smallvec![(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), true)]
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutData<'a> for &'a mut T {
    type Item = (EntityId, &'a mut T);
    type Chunk = (&'a [EntityId], &'a mut [T]);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::FetchMut<'a> for &'a mut T {
    type State = std::slice::IterMut<'a, T>;

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare(
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        unsafe { <Self as sealed::ParamFetch<'a>>::prepare(desc, range) }
    }

    #[inline(always)]
    fn fast_prepare_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let state = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_prepare(
                archetype.component_types_slice(),
                archetype.flat_columns(),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        archetype.bump_columns_version_in_place();
        Some((state, entities))
    }

    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<(Self::State, &'a [EntityId])> {
        if let Some(&slot) = col_slots.first() {
            let len = archetype.len();
            let state = unsafe {
                <Self as sealed::ParamFetch<'a>>::fast_prepare_slot(
                    archetype.component_types_slice(),
                    archetype.flat_columns(),
                    slot,
                    len,
                )?
            };
            let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
            archetype.bump_columns_version_in_place();
            return Some((state, entities));
        }
        Self::fast_prepare_archetype(archetype)
    }

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        (
            entity,
            state
                .next()
                .expect("validated equal-length component and entity slices"),
        )
    }

    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <Self as sealed::ParamFetch<'a>>::skip_items(state, count);
    }

    #[inline]
    fn chunk(
        entities: &'a [EntityId],
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        Some((entities, unsafe { <Self as sealed::ParamFetch<'a>>::chunk(desc, range)? }))
    }

    #[inline(always)]
    fn fast_chunk_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let chunk = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_chunk(
                archetype.component_types_slice(),
                archetype.flat_columns(),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        archetype.bump_columns_version_in_place();
        Some((entities, chunk))
    }

    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        if let Some(&slot) = col_slots.first() {
            let len = archetype.len();
            let chunk = unsafe {
                <Self as sealed::ParamFetch<'a>>::fast_chunk_slot(
                    archetype.component_types_slice(),
                    archetype.flat_columns(),
                    slot,
                    len,
                )?
            };
            let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
            archetype.bump_columns_version_in_place();
            return Some((entities, chunk));
        }
        Self::fast_chunk_archetype(archetype)
    }

    #[inline]
    fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
        smallvec::smallvec![(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), true)]
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutData<'a> for &'a T {
    type Item = (EntityId, &'a T);
    type Chunk = (&'a [EntityId], &'a [T]);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::FetchMut<'a> for &'a T {
    type State = std::slice::Iter<'a, T>;

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare(
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        unsafe { <Self as sealed::ParamFetch<'a>>::prepare(desc, range) }
    }

    #[inline(always)]
    fn fast_prepare_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let state = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_prepare(
                archetype.component_types_slice(),
                archetype.flat_columns(),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((state, entities))
    }

    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<(Self::State, &'a [EntityId])> {
        if let Some(&slot) = col_slots.first() {
            let len = archetype.len();
            let state = unsafe {
                <Self as sealed::ParamFetch<'a>>::fast_prepare_slot(
                    archetype.component_types_slice(),
                    archetype.flat_columns(),
                    slot,
                    len,
                )?
            };
            let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
            return Some((state, entities));
        }
        Self::fast_prepare_archetype(archetype)
    }

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        (
            entity,
            state.next().expect("validated equal-length component slice"),
        )
    }

    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <Self as sealed::ParamFetch<'a>>::skip_items(state, count);
    }

    #[inline]
    fn chunk(
        entities: &'a [EntityId],
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        Some((entities, unsafe { <Self as sealed::ParamFetch<'a>>::chunk(desc, range)? }))
    }

    #[inline(always)]
    fn fast_chunk_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let chunk = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_chunk(
                archetype.component_types_slice(),
                archetype.flat_columns(),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((entities, chunk))
    }

    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        if let Some(&slot) = col_slots.first() {
            let len = archetype.len();
            let chunk = unsafe {
                <Self as sealed::ParamFetch<'a>>::fast_chunk_slot(
                    archetype.component_types_slice(),
                    archetype.flat_columns(),
                    slot,
                    len,
                )?
            };
            let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
            return Some((entities, chunk));
        }
        Self::fast_chunk_archetype(archetype)
    }

    #[inline(always)]
    fn mark_written_columns(
        _archetype: &mut Archetype,
        _masked: Option<&RowBitMask>,
        _epoch: u64,
    ) {
    }

    #[inline]
    fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
        smallvec::smallvec![(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), false)]
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutData<'a> for Option<&'a T> {
    type Item = (EntityId, Option<&'a T>);
    type Chunk = (&'a [EntityId], Option<&'a [T]>);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::FetchMut<'a> for Option<&'a T> {
    type State = Option<std::slice::Iter<'a, T>>;

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn prepare(
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<Self::State> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        unsafe { <Self as sealed::ParamFetch<'a>>::prepare(desc, range) }
    }

    #[inline(always)]
    fn fast_prepare_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let state = unsafe { <Self as sealed::ParamFetch<'a>>::fast_prepare(comp_slice, flat_cols, len)? };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((state, entities))
    }

    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<(Self::State, &'a [EntityId])> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let state = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_prepare_slot(
                comp_slice,
                flat_cols,
                col_slots.first().copied().unwrap_or(u16::MAX),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((state, entities))
    }

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        (entity, <Self as sealed::ParamFetch<'a>>::next_item(state))
    }

    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        <Self as sealed::ParamFetch<'a>>::skip_items(state, count);
    }

    #[inline]
    fn chunk(
        entities: &'a [EntityId],
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let cid = ComponentTypeId::of::<T>();
        let desc = columns
            .first()
            .filter(|(id, _)| *id == cid)
            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
            .map(|(_, d)| d);
        let c = unsafe { <Self as sealed::ParamFetch<'a>>::chunk(desc, range)? };
        Some((entities, c))
    }

    #[inline(always)]
    fn fast_chunk_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let chunk = unsafe { <Self as sealed::ParamFetch<'a>>::fast_chunk(comp_slice, flat_cols, len)? };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((entities, chunk))
    }

    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        let len = archetype.len();
        let comp_slice = archetype.component_types_slice();
        let flat_cols = archetype.flat_columns();
        let chunk = unsafe {
            <Self as sealed::ParamFetch<'a>>::fast_chunk_slot(
                comp_slice,
                flat_cols,
                col_slots.first().copied().unwrap_or(u16::MAX),
                len,
            )?
        };
        let entities = unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
        Some((entities, chunk))
    }

    #[inline(always)]
    fn mark_written_columns(
        _archetype: &mut Archetype,
        _masked: Option<&RowBitMask>,
        _epoch: u64,
    ) {
    }

    #[inline]
    fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
        smallvec::smallvec![(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), false)]
    }
}

macro_rules! tuple_query {
    ($($name:ident:$index:tt),+) => {
        #[allow(private_bounds)]
        impl<'a, $($name: QueryMutParam<'a> + sealed::ReadParam<'a>),+> QueryData<'a>
            for ($($name,)+)
        {
            type Item = (EntityId, ($(<$name as QueryMutParam<'a>>::Item,)+));
            type Chunk = (&'a [EntityId], ($(<$name as QueryMutParam<'a>>::Chunk,)+));
        }

        impl<'a, $($name: QueryMutParam<'a> + sealed::ReadParam<'a>),+> sealed::Fetch<'a>
            for ($($name,)+)
        {
            type State = ($(<$name as sealed::ReadParam<'a>>::ReadState,)+);
            #[inline]
            fn total_row_width() -> usize {
                $(<$name as sealed::ReadParam<'a>>::element_size() +)+ 0
            }
            #[inline]
            fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
                let cols = ($(
                    $name::prepare_read(archetype, start)?,
                )+);
                let mut end = archetype.len();
                $(
                    end = end.min(cols.$index.1);
                )+
                Some((($(cols.$index.0,)+), end))
            }
            #[inline(always)]
            fn fast_prepare_bound(
                archetype: &'a Archetype,
                col_slots: &[u16],
                start: usize,
            ) -> Option<(Self::State, usize)> {
                let cols = ($(
                    $name::fast_prepare_read_slot(
                        archetype,
                        col_slots.get($index).copied().unwrap_or(u16::MAX),
                        start,
                    )?,
                )+);
                let mut end = archetype.len();
                $(
                    end = end.min(cols.$index.1);
                )+
                Some((($(cols.$index.0,)+), end))
            }
            #[inline]
            fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
                (entity, ($($name::next_read(&mut state.$index),)+))
            }
            #[inline]
            fn skip_rows(state: &mut Self::State, count: usize) {
                $(
                    $name::skip_read(&mut state.$index, count);
                )+
            }
            #[inline]
            fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
                let slices = ($($name::chunk_read(archetype, start)?,)+);
                let len = archetype.len() - start;
                $(let len = len.min(slices.$index.1);)+
                let end = start + len;
                Some((
                    (&archetype.entities()[start..end], ($($name::truncate_chunk(slices.$index.0, len),)+)),
                    end,
                ))
            }
            #[inline(always)]
            fn fast_chunk_bound(
                archetype: &'a Archetype,
                col_slots: &[u16],
                start: usize,
            ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
                let slices = ($(
                    $name::fast_chunk_read_slot(
                        archetype,
                        col_slots.get($index).copied().unwrap_or(u16::MAX),
                        start,
                    )?,
                )+);
                let len = archetype.len() - start;
                $(let len = len.min(slices.$index.1);)+
                let end = start + len;
                Some((
                    (&archetype.entities()[start..end], ($($name::truncate_chunk(slices.$index.0, len),)+)),
                    end,
                ))
            }
        }
    };
}

macro_rules! tuple_query_mut {
    ($($P:ident : $idx:tt),+) => {
        impl<'a, $($P: QueryMutParam<'a>),+> QueryMutData<'a> for ($($P,)+) {
            type Item = (EntityId, ($($P::Item,)+));
            type Chunk = (&'a [EntityId], ($($P::Chunk,)+));

            fn validate_query() {
                let ids = [$($P::component_type_id()),+];
                let type_names = [$($P::type_name()),+];
                for i in 0..ids.len() {
                    for j in (i + 1)..ids.len() {
                        assert_ne!(
                            ids[i], ids[j],
                            "Aliasing violation: component '{}' requested multiple times in query_mut",
                            type_names[i]
                        );
                    }
                }
            }
        }

        impl<'a, $($P: QueryMutParam<'a>),+> sealed::FetchMut<'a> for ($($P,)+) {
            type State = ($($P::State,)+);

            #[inline]
            fn total_row_width() -> usize {
                $(<$P as sealed::ParamFetch<'a>>::element_size() +)+ 0
            }

            #[inline]
            #[allow(non_snake_case)]
            fn prepare(
                columns: &[(ComponentTypeId, ColumnDescriptor)],
                range: Range<usize>,
            ) -> Option<Self::State> {
                Some(($(
                    unsafe {
                        let cid = $P::component_type_id();
                        let desc = columns
                            .get($idx)
                            .filter(|(id, _)| *id == cid)
                            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
                            .map(|(_, d)| d);
                        $P::prepare(desc, range.clone())?
                    },
                )+))
            }

            #[inline(always)]
            fn fast_prepare_archetype(
                archetype: &'a mut Archetype,
            ) -> Option<(Self::State, &'a [EntityId])> {
                let len = archetype.len();
                let comp_slice = archetype.component_types_slice();
                let flat_cols = archetype.flat_columns();
                let state = ($(
                    unsafe { $P::fast_prepare(comp_slice, flat_cols, len)? },
                )+);
                let entities =
                    unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
                if $($P::is_write())||+ {
                    archetype.bump_columns_version_in_place();
                }
                Some((state, entities))
            }

            #[inline(always)]
            fn fast_prepare_bound(
                archetype: &'a mut Archetype,
                col_slots: &[u16],
            ) -> Option<(Self::State, &'a [EntityId])> {
                let len = archetype.len();
                let comp_slice = archetype.component_types_slice();
                let flat_cols = archetype.flat_columns();
                let state = ($(
                    unsafe {
                        $P::fast_prepare_slot(
                            comp_slice,
                            flat_cols,
                            col_slots.get($idx).copied().unwrap_or(u16::MAX),
                            len,
                        )?
                    },
                )+);
                let entities =
                    unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
                if $($P::is_write())||+ {
                    archetype.bump_columns_version_in_place();
                }
                Some((state, entities))
            }

            #[inline]
            fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
                (entity, ($($P::next_item(&mut state.$idx),)+))
            }

            #[inline]
            fn skip_rows(state: &mut Self::State, count: usize) {
                $(
                    $P::skip_items(&mut state.$idx, count);
                )+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn chunk(
                entities: &'a [EntityId],
                columns: &[(ComponentTypeId, ColumnDescriptor)],
                range: Range<usize>,
            ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
                Some((entities, ($(
                    unsafe {
                        let cid = $P::component_type_id();
                        let desc = columns
                            .get($idx)
                            .filter(|(id, _)| *id == cid)
                            .or_else(|| columns.iter().find(|(id, _)| *id == cid))
                            .map(|(_, d)| d);
                        $P::chunk(desc, range.clone())?
                    },
                )+)))
            }

            #[inline(always)]
            fn fast_chunk_archetype(
                archetype: &'a mut Archetype,
            ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
                let len = archetype.len();
                let comp_slice = archetype.component_types_slice();
                let flat_cols = archetype.flat_columns();
                let chunks = ($(
                    unsafe { $P::fast_chunk(comp_slice, flat_cols, len)? },
                )+);
                let entities =
                    unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
                if $($P::is_write())||+ {
                    archetype.bump_columns_version_in_place();
                }
                Some((entities, chunks))
            }

            #[inline(always)]
            fn fast_chunk_bound(
                archetype: &'a mut Archetype,
                col_slots: &[u16],
            ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
                let len = archetype.len();
                let comp_slice = archetype.component_types_slice();
                let flat_cols = archetype.flat_columns();
                let chunks = ($(
                    unsafe {
                        $P::fast_chunk_slot(
                            comp_slice,
                            flat_cols,
                            col_slots.get($idx).copied().unwrap_or(u16::MAX),
                            len,
                        )?
                    },
                )+);
                let entities =
                    unsafe { std::slice::from_raw_parts(archetype.entities().as_ptr(), len) };
                if $($P::is_write())||+ {
                    archetype.bump_columns_version_in_place();
                }
                Some((entities, chunks))
            }

            #[inline]
            fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
                smallvec::smallvec![
                    $(
                        ($P::component_type_id(), $P::type_name(), $P::is_write()),
                    )+
                ]
            }

            #[inline]
            fn required_mask() -> u16 {
                0u16 $(| (($P::IS_REQUIRED as u16) << $idx))+
            }
        }
    };
}

macro_rules! all_tuples {
    ($mac:ident) => {
        $mac!(A:0);
        $mac!(A:0, B:1);
        $mac!(A:0, B:1, C:2);
        $mac!(A:0, B:1, C:2, D:3);
        $mac!(A:0, B:1, C:2, D:3, E:4);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14);
        $mac!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15);
    };
}

all_tuples!(tuple_query);
all_tuples!(tuple_query_mut);


#[path = "iter/mod.rs"]
pub mod iter;
pub use iter::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
