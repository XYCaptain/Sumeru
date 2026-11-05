//! Borrowed, allocation-free queries over archetype-local component columns.

use crate::storage::archetype::Archetype;
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
/// impl<'a> sumer_state::QueryData<'a> for ExternalQuery {
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
/// impl<'a> sumer_state::QueryMutData<'a> for ExternalMutQuery {
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
    pub(crate) fn new(previous: F) -> Self {
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
    pub(crate) fn new(previous: F) -> Self {
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

impl<F: sealed::Filter + Send + Sync + 'static> QueryFilter for F {}

pub(crate) mod sealed {
    use super::*;

    pub(crate) trait Fetch<'a> {
        type State;
        fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)>;
        fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item
        where
            Self: QueryData<'a>;
        fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)>
        where
            Self: QueryData<'a>;
    }

    pub(crate) trait FetchMut<'a> {
        type State;
        fn prepare(
            columns: &[(ComponentTypeId, ColumnDescriptor)],
            range: Range<usize>,
        ) -> Option<Self::State>;
        fn fast_prepare_archetype(
            archetype: &'a mut Archetype,
        ) -> Option<(Self::State, &'a [EntityId])>;
        fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item
        where
            Self: QueryMutData<'a>;
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
        #[allow(dead_code)]
        fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]>;
        #[inline]
        fn required_mask() -> u16 {
            u16::MAX
        }
    }

    pub(crate) trait ReadParam<'a>: QueryMutParam<'a> {
        type ReadState: 'a;
        fn prepare_read(archetype: &'a Archetype, start: usize) -> Option<(Self::ReadState, usize)>;
        fn next_read(state: &mut Self::ReadState) -> <Self as QueryMutParam<'a>>::Item;
        fn chunk_read(
            archetype: &'a Archetype,
            start: usize,
        ) -> Option<(<Self as QueryMutParam<'a>>::Chunk, usize)>;
        fn truncate_chunk(
            chunk: <Self as QueryMutParam<'a>>::Chunk,
            len: usize,
        ) -> <Self as QueryMutParam<'a>>::Chunk;
    }

    pub(crate) trait ParamFetch<'a> {
        type State: 'a;
        const IS_REQUIRED: bool = true;
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
        fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item
        where
            Self: QueryMutParam<'a>;
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
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            if let Some(state) = self.prepare_table(accessor) {
                self.matches_row_prepared(&state, row)
            } else {
                false
            }
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
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            self.previous.matches_row_internal(accessor, row)
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
        fn matches_row_internal<A: RowColumnAccessor + ?Sized>(&self, accessor: &A, row: usize) -> bool {
            self.previous.matches_row_internal(accessor, row)
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

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::ReadParam<'a> for &'a T {
    type ReadState = ColumnIter<'a, T>;

    #[inline]
    fn prepare_read(archetype: &'a Archetype, start: usize) -> Option<(Self::ReadState, usize)> {
        let col = archetype.get_column::<T>()?;
        if col.len() != archetype.len() {
            return None;
        }
        Some((col.iter_from(start), archetype.len()))
    }

    #[inline]
    fn next_read(state: &mut Self::ReadState) -> &'a T {
        state
            .next()
            .expect("validated equal-length component and entity slices")
    }

    #[inline]
    fn chunk_read(archetype: &'a Archetype, start: usize) -> Option<(&'a [T], usize)> {
        let slice = column_suffix::<T>(archetype, start)?;
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

    #[inline]
    fn next_read(state: &mut Self::ReadState) -> Option<&'a T> {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component and entity slices"))
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
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        <&'a T as sealed::ReadParam<'a>>::prepare_read(archetype, start)
    }
    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        (entity, <&'a T as sealed::ReadParam<'a>>::next_read(state))
    }
    #[inline]
    fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        let slice = column_suffix::<T>(archetype, start)?;
        let end = start + slice.len();
        Some(((&archetype.entities()[start..end], slice), end))
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryData<'a> for Option<&'a T> {
    type Item = (EntityId, Option<&'a T>);
    type Chunk = (&'a [EntityId], Option<&'a [T]>);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::Fetch<'a> for Option<&'a T> {
    type State = Option<ColumnIter<'a, T>>;
    #[inline]
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        <Option<&'a T> as sealed::ReadParam<'a>>::prepare_read(archetype, start)
    }
    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        (entity, <Option<&'a T> as sealed::ReadParam<'a>>::next_read(state))
    }
    #[inline]
    fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        let (opt_slice, avail) = <Option<&'a T> as sealed::ReadParam<'a>>::chunk_read(archetype, start)?;
        let end = start + avail;
        Some(((&archetype.entities()[start..end], opt_slice), end))
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

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .next()
            .expect("validated equal-length component slice")
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

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .next()
            .expect("validated equal-length component slice")
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

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component slice"))
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

    #[inline]
    fn next_item(state: &mut Self::State) -> <Self as QueryMutParam<'a>>::Item {
        state
            .as_mut()
            .map(|it| it.next().expect("validated equal-length component slice"))
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryMutData<'a> for &'a mut T {
    type Item = (EntityId, &'a mut T);
    type Chunk = (&'a [EntityId], &'a mut [T]);
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> sealed::FetchMut<'a> for &'a mut T {
    type State = std::slice::IterMut<'a, T>;

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

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        (
            entity,
            state.next().expect("validated equal-length component slice"),
        )
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
            #[inline]
            fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
                (entity, ($($name::next_read(&mut state.$index),)+))
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

            #[inline]
            fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
                (entity, ($($P::next_item(&mut state.$idx),)+))
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
