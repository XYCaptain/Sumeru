use super::*;
use crate::id::ArchetypeId;

/// 纯函数：根据给定的查询形态 Q 和过滤器 F，从原型头切片中筛选匹配的 ArchetypeId 列表。
pub(crate) fn match_archetypes_for_query<Q: QueryMutShape, F: QueryFilter, H: ArchetypeHeader>(
    archetypes: &[H],
    filter: &F,
) -> Result<Vec<ArchetypeId>, StateError> {
    Q::validate_shape()?;
    let descriptors = Q::access_descriptors();
    let mut matched = Vec::new();
    for arch in archetypes {
        if filter.matches_archetype(arch)
            && descriptors
                .iter()
                .all(|(cid, _, _)| arch.contains_component(*cid))
        {
            matched.push(arch.archetype_id());
        }
    }
    Ok(matched)
}

/// 共享的原型匹配算法：扫描 Store 中所有原型，按查询形态和过滤器筛选匹配的原型 ID。
///
/// 当代际（`archetype_set_generation`）不一致时，**全量重新扫描原型集合**并重建匹配列表。
/// 确保所有查询路径使用完全相同的匹配逻辑。
#[inline]
#[allow(dead_code)]
pub(crate) fn rebuild_matched_archetypes<Q: QueryMutShape, F: QueryFilter>(
    store: &Store,
    filter: &F,
) -> Vec<ArchetypeId> {
    match_archetypes_for_query::<Q, F, Archetype>(&store.archetypes, filter)
        .unwrap_or_default()
}

/// 纯函数：向已有匹配切片追加扫描新增原型切片（用于 `QueryCache` 的 $O(\Delta A)$ 增量刷新）。
pub(crate) fn append_bindings_from_slice<
    Q: QueryMutShape,
    F: QueryFilter,
    H: ArchetypeHeader,
>(
    archetypes: &[H],
    filter: &F,
    matched: &mut Vec<ArchetypeId>,
    bindings: &mut Vec<crate::storage::query::handle::MatchedTableBinding>,
) -> Result<(), StateError> {
    Q::validate_shape()?;
    let descriptors = Q::access_descriptors();
    for arch in archetypes {
        if !filter.matches_archetype(arch) {
            continue;
        }
        let mut flat_col_indices = smallvec::SmallVec::<[u16; 8]>::with_capacity(descriptors.len());
        let mut all_present = true;
        for &(cid, _, _) in &descriptors {
            if let Some(idx) = arch.component_column_index(cid) {
                flat_col_indices.push(idx);
            } else {
                all_present = false;
                break;
            }
        }
        if all_present {
            let archetype_id = arch.archetype_id();
            matched.push(archetype_id);
            bindings.push(crate::storage::query::handle::MatchedTableBinding {
                archetype_id,
                flat_col_indices,
            });
        }
    }
    Ok(())
}

/// 纯函数：根据给定的查询形态 Q 和过滤器 F，从原型头切片中筛选匹配的 ArchetypeId 列表并预绑定物理列槽位。
pub(crate) fn match_bindings_for_query<
    Q: QueryMutShape,
    F: QueryFilter,
    H: ArchetypeHeader,
>(
    archetypes: &[H],
    filter: &F,
) -> Result<(Vec<ArchetypeId>, Vec<crate::storage::query::handle::MatchedTableBinding>), StateError> {
    let mut matched = Vec::new();
    let mut bindings = Vec::new();
    append_bindings_from_slice::<Q, F, H>(archetypes, filter, &mut matched, &mut bindings)?;
    Ok((matched, bindings))
}

/// 利用 `Store::component_archetypes` 倒排位图索引对查询形态 `Q` 与过滤器 `F` 进行 64-bit word 位图求交，
/// 杜绝全量遍历无关 Archetype，并预绑定物理列下标。
pub(crate) fn match_bindings_for_store<Q: QueryMutShape, F: QueryFilter>(
    store: &Store,
    filter: &F,
) -> Result<(Vec<ArchetypeId>, Vec<crate::storage::query::handle::MatchedTableBinding>), StateError> {
    Q::validate_shape()?;
    if store.archetypes.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let descriptors = Q::access_descriptors();
    let mut reqs = Vec::with_capacity(descriptors.len() + 4);
    for &(cid, _, _) in &descriptors {
        reqs.push((cid, true));
    }
    filter.archetype_requirements(&mut reqs);

    let comp_index = store.component_archetypes();
    let mut candidate_bits: Option<crate::storage::store::ArchetypeBitSet> = None;
    let mut has_positive_req = false;

    for &(cid, must_have) in &reqs {
        if must_have {
            has_positive_req = true;
            let Some(bits) = comp_index.get(&cid) else {
                return Ok((Vec::new(), Vec::new()));
            };
            if let Some(acc) = &mut candidate_bits {
                acc.intersect_with(bits);
                if acc.is_all_zero() {
                    return Ok((Vec::new(), Vec::new()));
                }
            } else {
                candidate_bits = Some(bits.clone());
            }
        }
    }

    if !has_positive_req {
        return match_bindings_for_query::<Q, F, Archetype>(&store.archetypes, filter);
    }

    let Some(mut acc) = candidate_bits else {
        return Ok((Vec::new(), Vec::new()));
    };

    for &(cid, must_have) in &reqs {
        if !must_have {
            if let Some(bits) = comp_index.get(&cid) {
                acc.difference_with(bits);
                if acc.is_all_zero() {
                    return Ok((Vec::new(), Vec::new()));
                }
            }
        }
    }

    let mut matched = Vec::new();
    let mut bindings = Vec::new();
    acc.for_each_set_bit(|arch_idx| {
        let Some(arch) = store.archetypes.get(arch_idx as usize) else {
            return;
        };
        if !filter.matches_archetype(arch) {
            return;
        }
        let mut flat_col_indices = smallvec::SmallVec::<[u16; 8]>::with_capacity(descriptors.len());
        for &(cid, _, _) in &descriptors {
            let Some(idx) = arch.component_column_index(cid) else {
                return;
            };
            flat_col_indices.push(idx);
        }
        let archetype_id = arch.archetype_id();
        matched.push(archetype_id);
        bindings.push(crate::storage::query::handle::MatchedTableBinding {
            archetype_id,
            flat_col_indices,
        });
    });

    Ok((matched, bindings))
}

#[inline]
pub(crate) fn rebuild_matched_bindings<Q: QueryMutShape, F: QueryFilter>(
    store: &Store,
    filter: &F,
) -> (Vec<ArchetypeId>, Vec<crate::storage::query::handle::MatchedTableBinding>) {
    match_bindings_for_store::<Q, F>(store, filter).unwrap_or_default()
}

/// 合并查询形状 `Q` 的读写列与过滤器 `F` 读取列的前置权限描述符列表。
///
/// 若某组件在 `Q` 中为写权限（`is_write = true`），即使在 `filter` 中仅作为读取，其权限依然保持写权限；
/// 若某组件在 `Q` 中为读权限或仅在 `filter` 中出现，则登记为读权限（`is_write = false`）。
pub fn merged_access_descriptors<Q: QueryMutShape, F: QueryFilter>(
    filter: &F,
) -> Vec<(ComponentTypeId, &'static str, bool)> {
    let mut map = std::collections::BTreeMap::new();
    Q::for_each_access_descriptor(|cid, name, is_write| {
        map.insert(cid, (name, is_write));
    });
    QueryFilter::filter_read_components(filter, &mut |cid, name| {
        map.entry(cid)
            .and_modify(|(_, _is_write)| {
                // If it's already write, it stays write!
            })
            .or_insert((name, false));
    });
    map.into_iter().map(|(cid, (name, write))| (cid, name, write)).collect()
}

/// 强类型原生列可变指针包装器（根据组件的 Send 实现跨线程安全发送）
#[derive(Debug, Clone)]
pub struct MutColumnPtr<T>(ColumnDescriptor, PhantomData<T>);
unsafe impl<T: Send> Send for MutColumnPtr<T> {}

/// 强类型原生列只读指针包装器（根据组件的 Sync 实现跨线程安全发送）
#[derive(Debug, Clone)]
pub struct ConstColumnPtr<T>(ColumnDescriptor, PhantomData<T>);
unsafe impl<T: Sync> Send for ConstColumnPtr<T> {}

/// Represents the static shape of a mutable query parameter, decoupled from borrow lifetimes.
///
/// This trait is sealed: only internal shapes (`&'static T` and `&'static mut T`) may implement it.
///
/// An external type cannot implement the sealed `QueryMutParamShape` trait because
/// the supertrait `sealed::QueryMutParamShapeInternal` is private to this crate:
/// ```compile_fail,E0277
/// struct ExternalParamShape;
/// impl sumeru::QueryMutParamShape for ExternalParamShape {
///     fn component_type_id() -> sumeru::ComponentTypeId { sumeru::ComponentTypeId::of::<i32>() }
///     fn type_name() -> &'static str { "i32" }
///     fn is_write() -> bool { true }
/// }
/// ```
#[allow(private_bounds, private_interfaces)]
pub trait QueryMutParamShape: sealed::QueryMutParamShapeInternal + Send + 'static {
    fn component_type_id() -> ComponentTypeId;
    fn type_name() -> &'static str;
    fn is_write() -> bool;
    #[inline]
    fn element_size() -> usize {
        0
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned + Send> QueryMutParamShape
    for &'static mut T
{
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
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> sealed::QueryMutParamShapeInternal
    for &'static mut T
{
    type Param<'a> = &'a mut T;
    type ColumnBuffer = MutColumnPtr<T>;
    type IterState<'a> = std::slice::IterMut<'a, T>;
    #[inline]
    fn segment_end_with_cursor(buffer: &Self::ColumnBuffer, row: usize, cursor: &mut usize) -> usize {
        buffer.0.locate_with_cursor(row, cursor).2
    }
    #[inline]
    #[allow(private_interfaces)]
    unsafe fn resolve_buffer(
        desc: &ColumnDescriptor,
        expected_len: usize,
    ) -> Option<MutColumnPtr<T>> {
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) || desc.len != expected_len {
            return None;
        }
        Some(MutColumnPtr(desc.clone(), PhantomData))
    }
    #[inline]
    unsafe fn prepare_state<'view>(
        buffer: &MutColumnPtr<T>,
        range: Range<usize>,
        cursor: &mut usize,
    ) -> std::slice::IterMut<'view, T> {
        let (base, offset, end) = buffer.0.locate_with_cursor(range.start, cursor);
        assert!(range.end <= end);
        let ptr = unsafe { (base as *mut T).add(offset) };
        unsafe { std::slice::from_raw_parts_mut(ptr, range.len()).iter_mut() }
    }
    #[inline]
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
    ) -> <Self::Param<'view> as QueryMutParam<'view>>::Item {
        debug_assert!(state.len() != 0, "validated equal-length component slice");
        // SAFETY: this sealed operation requires an unexhausted iterator.
        // QueryView prepares equal-length component iterators for one exact
        // selected run and advances components only after the equally sized
        // identity iterator has yielded an item.
        unsafe { state.next().unwrap_unchecked() }
    }
    #[inline]
    unsafe fn skip_items<'view>(state: &mut Self::IterState<'view>, count: usize) {
        if count > 0 {
            let _ = state.nth(count - 1);
        }
    }
    #[inline]
    unsafe fn chunk_state<'view>(buffer: &MutColumnPtr<T>, range: Range<usize>, cursor: &mut usize) -> &'view mut [T] {
        let (base, offset, end) = buffer.0.locate_with_cursor(range.start, cursor);
        assert!(range.end <= end);
        let ptr = unsafe { (base as *mut T).add(offset) };
        unsafe { std::slice::from_raw_parts_mut(ptr, range.len()) }
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned + Sync> QueryMutParamShape for &'static T {
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
    fn element_size() -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> sealed::QueryMutParamShapeInternal
    for &'static T
{
    type Param<'a> = &'a T;
    type ColumnBuffer = ConstColumnPtr<T>;
    type IterState<'a> = std::slice::Iter<'a, T>;
    #[inline]
    fn segment_end_with_cursor(buffer: &Self::ColumnBuffer, row: usize, cursor: &mut usize) -> usize {
        buffer.0.locate_with_cursor(row, cursor).2
    }
    #[inline]
    #[allow(private_interfaces)]
    unsafe fn resolve_buffer(
        desc: &ColumnDescriptor,
        expected_len: usize,
    ) -> Option<ConstColumnPtr<T>> {
        if desc.rust_type_id != Some(std::any::TypeId::of::<T>()) || desc.len != expected_len {
            return None;
        }
        Some(ConstColumnPtr(desc.clone(), PhantomData))
    }
    #[inline]
    unsafe fn prepare_state<'view>(
        buffer: &ConstColumnPtr<T>,
        range: Range<usize>,
        cursor: &mut usize,
    ) -> std::slice::Iter<'view, T> {
        let (base, offset, end) = buffer.0.locate_with_cursor(range.start, cursor);
        assert!(range.end <= end);
        let ptr = unsafe { (base as *const T).add(offset) };
        unsafe { std::slice::from_raw_parts(ptr, range.len()).iter() }
    }
    #[inline]
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
    ) -> <Self::Param<'view> as QueryMutParam<'view>>::Item {
        debug_assert!(state.len() != 0, "validated equal-length component slice");
        // SAFETY: the same unexhausted-run contract as the mutable parameter.
        unsafe { state.next().unwrap_unchecked() }
    }
    #[inline]
    unsafe fn skip_items<'view>(state: &mut Self::IterState<'view>, count: usize) {
        if count > 0 {
            let _ = state.nth(count - 1);
        }
    }
    #[inline]
    unsafe fn chunk_state<'view>(buffer: &ConstColumnPtr<T>, range: Range<usize>, cursor: &mut usize) -> &'view [T] {
        let (base, offset, end) = buffer.0.locate_with_cursor(range.start, cursor);
        assert!(range.end <= end);
        let ptr = unsafe { (base as *const T).add(offset) };
        unsafe { std::slice::from_raw_parts(ptr, range.len()) }
    }
}

/// Represents the static shape of a mutable query, decoupled from any specific borrow lifetime.
#[allow(private_bounds)]
pub trait QueryMutShape:
    sealed::QueryMutShapeInternal<Self::ArchetypeBuffers> + Send + 'static
{
    type View<'view>: QueryMutData<'view, Item = Self::Item<'view>, Chunk = Self::Chunk<'view>>;
    type Item<'view>;
    type Chunk<'view>;
    type ArchetypeBuffers: Send + Clone;
    type IterState<'view>;
    type Cursor: Default + Send + 'static;

    const PARAM_COUNT: usize;
    fn for_each_access_descriptor(f: impl FnMut(ComponentTypeId, &'static str, bool));

    #[inline]
    fn access_descriptors() -> Vec<(ComponentTypeId, &'static str, bool)> {
        let mut v = Vec::with_capacity(Self::PARAM_COUNT);
        Self::for_each_access_descriptor(|cid, name, write| v.push((cid, name, write)));
        v
    }

    /// 校验查询形状内部是否存在别名/读写冲突（例如元组中对同一组件同时存在写操作）
    fn validate_shape() -> Result<(), StateError>;

    /// 计算该查询形状中所有组件列的总行宽（字节）
    #[inline]
    fn total_row_width() -> usize {
        0
    }

    /// 计算当前游标所在分段边界
    fn segment_end_with_cursor(
        buffers: &Self::ArchetypeBuffers,
        row: usize,
        cursor: &mut Self::Cursor,
    ) -> usize;

    /// 迭代准备：直接使用扁平物理基址与区间，零虚调用，O(1) 定位
    unsafe fn prepare_iter<'view>(
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::IterState<'view>;

    /// 行迭代推进。
    ///
    /// # Safety
    /// All component iterators must have at least one remaining item, prepared
    /// for the same exact leased run. Each call advances every component once;
    /// callers must stop at the run length before preparing another disjoint run.
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
        entity: EntityId,
    ) -> Self::Item<'view>;

    /// 跳过指定行数推进迭代器（用于行级过滤）
    unsafe fn skip_items<'view>(
        state: &mut Self::IterState<'view>,
        count: usize,
    );

    /// 切片分块提取
    unsafe fn chunk<'view>(
        entities: &'view [EntityId],
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::Chunk<'view>;
}

macro_rules! count_params {
    () => { 0 };
    ($head:ident $($tail:ident)*) => { 1 + count_params!($($tail)*) };
}

macro_rules! tuple_query_mut_shape {
    ($($P:ident : $idx:tt),+) => {
        impl<$($P: QueryMutParamShape),+> sealed::QueryMutShapeInternal<($($P::ColumnBuffer,)+)> for ($($P,)+) {
            #[inline]
            unsafe fn resolve_archetype_buffers(
                get_desc: &dyn Fn(ComponentTypeId) -> Option<ColumnDescriptor>,
                expected_len: usize,
            ) -> Option<($($P::ColumnBuffer,)+)> {
                Some((
                    $(
                        unsafe {
                            let desc = get_desc($P::component_type_id())?;
                            $P::resolve_buffer(&desc, expected_len)?
                        },
                    )+
                ))
            }
        }

        impl<$($P: QueryMutParamShape),+> QueryMutShape for ($($P,)+) {
            type View<'view> = ($($P::Param<'view>,)+);
            type Item<'view> = (EntityId, ($(<$P::Param<'view> as QueryMutParam<'view>>::Item,)+));
            type Chunk<'view> = (&'view [EntityId], ($(<$P::Param<'view> as QueryMutParam<'view>>::Chunk,)+));
            type ArchetypeBuffers = ($($P::ColumnBuffer,)+);
            type IterState<'view> = ($($P::IterState<'view>,)+);
            type Cursor = [usize; 16];

            const PARAM_COUNT: usize = count_params!($($P)+);

            #[inline]
            fn for_each_access_descriptor(mut f: impl FnMut(ComponentTypeId, &'static str, bool)) {
                $(
                    f($P::component_type_id(), $P::type_name(), $P::is_write());
                )+
            }

            fn validate_shape() -> Result<(), StateError> {
                let descriptors = [
                    $(
                        ($P::component_type_id(), $P::type_name(), $P::is_write()),
                    )+
                ];
                for i in 0..descriptors.len() {
                    let (id_i, name_i, write_i) = descriptors[i];
                    for j in (i + 1)..descriptors.len() {
                        let (id_j, _, write_j) = descriptors[j];
                        if id_i == id_j && (write_i || write_j) {
                            return Err(StateError::SelfBorrowConflict {
                                component: name_i.to_string(),
                                detail: format!(
                                    "Conflicting tuple parameters: {} vs {}",
                                    if write_i { "Write" } else { "Read" },
                                    if write_j { "Write" } else { "Read" },
                                ),
                            });
                        }
                    }
                }
                Ok(())
            }

            #[inline]
            fn total_row_width() -> usize {
                $($P::element_size() +)+ 0
            }

            #[inline]
            fn segment_end_with_cursor(buffers: &Self::ArchetypeBuffers, row: usize, cursor: &mut Self::Cursor) -> usize {
                usize::MAX $(.min($P::segment_end_with_cursor(&buffers.$idx, row, &mut cursor[$idx])))+
            }

            #[inline]
            unsafe fn prepare_iter<'view>(
                buffers: &Self::ArchetypeBuffers,
                range: Range<usize>,
                cursor: &mut Self::Cursor,
            ) -> Self::IterState<'view> {
                (
                    $(
                        unsafe { $P::prepare_state(&buffers.$idx, range.clone(), &mut cursor[$idx]) },
                    )+
                )
            }

            #[inline]
            unsafe fn next_item<'view>(
                state: &mut Self::IterState<'view>,
                entity: EntityId,
            ) -> Self::Item<'view> {
                (
                    entity,
                    (
                        $(
                            unsafe { $P::next_item(&mut state.$idx) },
                        )+
                    ),
                )
            }

            #[inline]
            unsafe fn skip_items<'view>(
                state: &mut Self::IterState<'view>,
                count: usize,
            ) {
                $(
                    unsafe { $P::skip_items(&mut state.$idx, count) };
                )+
            }

            #[inline]
            unsafe fn chunk<'view>(
                entities: &'view [EntityId],
                buffers: &Self::ArchetypeBuffers,
                range: Range<usize>,
                cursor: &mut Self::Cursor,
            ) -> Self::Chunk<'view> {
                (
                    entities,
                    (
                        $(
                            unsafe { $P::chunk_state(&buffers.$idx, range.clone(), &mut cursor[$idx]) },
                        )+
                    ),
                )
            }
        }
    };
}

all_tuples!(tuple_query_mut_shape);

impl<T: ModelComponent + Serialize + DeserializeOwned + Send>
    sealed::QueryMutShapeInternal<MutColumnPtr<T>> for &'static mut T
{
    #[inline]
    unsafe fn resolve_archetype_buffers(
        get_desc: &dyn Fn(ComponentTypeId) -> Option<ColumnDescriptor>,
        expected_len: usize,
    ) -> Option<MutColumnPtr<T>> {
        unsafe {
            let desc = get_desc(ComponentTypeId::of::<T>())?;
            <&'static mut T as sealed::QueryMutParamShapeInternal>::resolve_buffer(
                &desc,
                expected_len,
            )
        }
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned + Send> QueryMutShape for &'static mut T {
    type View<'view> = &'view mut T;
    type Item<'view> = (EntityId, &'view mut T);
    type Chunk<'view> = (&'view [EntityId], &'view mut [T]);
    type ArchetypeBuffers = MutColumnPtr<T>;
    type IterState<'view> = std::slice::IterMut<'view, T>;
    type Cursor = usize;

    const PARAM_COUNT: usize = 1;

    #[inline]
    fn for_each_access_descriptor(mut f: impl FnMut(ComponentTypeId, &'static str, bool)) {
        f(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), true);
    }

    #[inline]
    fn validate_shape() -> Result<(), StateError> {
        Ok(())
    }

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn segment_end_with_cursor(buffer: &Self::ArchetypeBuffers, row: usize, cursor: &mut Self::Cursor) -> usize {
        <Self as sealed::QueryMutParamShapeInternal>::segment_end_with_cursor(buffer, row, cursor)
    }

    #[inline]
    unsafe fn prepare_iter<'view>(
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::IterState<'view> {
        unsafe {
            <&'static mut T as sealed::QueryMutParamShapeInternal>::prepare_state(buffers, range, cursor)
        }
    }

    #[inline]
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
        entity: EntityId,
    ) -> Self::Item<'view> {
        (entity, unsafe {
            <&'static mut T as sealed::QueryMutParamShapeInternal>::next_item(state)
        })
    }

    #[inline]
    unsafe fn skip_items<'view>(
        state: &mut Self::IterState<'view>,
        count: usize,
    ) {
        unsafe { <&'static mut T as sealed::QueryMutParamShapeInternal>::skip_items(state, count) }
    }

    #[inline]
    unsafe fn chunk<'view>(
        entities: &'view [EntityId],
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::Chunk<'view> {
        (entities, unsafe {
            <&'static mut T as sealed::QueryMutParamShapeInternal>::chunk_state(buffers, range, cursor)
        })
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned>
    sealed::QueryMutShapeInternal<ConstColumnPtr<T>> for &'static T
{
    #[inline]
    unsafe fn resolve_archetype_buffers(
        get_desc: &dyn Fn(ComponentTypeId) -> Option<ColumnDescriptor>,
        expected_len: usize,
    ) -> Option<ConstColumnPtr<T>> {
        let desc = get_desc(ComponentTypeId::of::<T>())?;
        // SAFETY: forwarded layout and lease requirements are unchanged.
        unsafe { <Self as sealed::QueryMutParamShapeInternal>::resolve_buffer(&desc, expected_len) }
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> QueryMutShape for &'static T {
    type View<'view> = &'view T;
    type Item<'view> = (EntityId, &'view T);
    type Chunk<'view> = (&'view [EntityId], &'view [T]);
    type ArchetypeBuffers = ConstColumnPtr<T>;
    type IterState<'view> = std::slice::Iter<'view, T>;
    type Cursor = usize;
    const PARAM_COUNT: usize = 1;

    fn for_each_access_descriptor(mut f: impl FnMut(ComponentTypeId, &'static str, bool)) {
        f(ComponentTypeId::of::<T>(), std::any::type_name::<T>(), false);
    }

    fn validate_shape() -> Result<(), StateError> {
        Ok(())
    }

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    fn segment_end_with_cursor(buffer: &Self::ArchetypeBuffers, row: usize, cursor: &mut Self::Cursor) -> usize {
        <Self as sealed::QueryMutParamShapeInternal>::segment_end_with_cursor(buffer, row, cursor)
    }

    #[inline]
    unsafe fn prepare_iter<'view>(
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::IterState<'view> {
        // SAFETY: caller guarantees a valid, read-leased range.
        unsafe { <Self as sealed::QueryMutParamShapeInternal>::prepare_state(buffers, range, cursor) }
    }

    #[inline]
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
        entity: EntityId,
    ) -> Self::Item<'view> {
        (entity, unsafe { <Self as sealed::QueryMutParamShapeInternal>::next_item(state) })
    }

    #[inline]
    unsafe fn skip_items<'view>(
        state: &mut Self::IterState<'view>,
        count: usize,
    ) {
        unsafe { <Self as sealed::QueryMutParamShapeInternal>::skip_items(state, count) }
    }

    #[inline]
    unsafe fn chunk<'view>(
        entities: &'view [EntityId],
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::Chunk<'view> {
        (entities, unsafe { <Self as sealed::QueryMutParamShapeInternal>::chunk_state(buffers, range, cursor) })
    }
}

impl<Q: QueryMutShape> sealed::QueryMutShapeInternal<Q::ArchetypeBuffers> for (EntityId, Q) {
    #[inline]
    unsafe fn resolve_archetype_buffers(
        get_desc: &dyn Fn(ComponentTypeId) -> Option<ColumnDescriptor>,
        expected_len: usize,
    ) -> Option<Q::ArchetypeBuffers> {
        unsafe { Q::resolve_archetype_buffers(get_desc, expected_len) }
    }
}

impl<Q: QueryMutShape> QueryMutShape for (EntityId, Q) {
    type View<'view> = (EntityId, Q::View<'view>);
    type Item<'view> = Q::Item<'view>;
    type Chunk<'view> = Q::Chunk<'view>;
    type ArchetypeBuffers = Q::ArchetypeBuffers;
    type IterState<'view> = Q::IterState<'view>;
    type Cursor = Q::Cursor;

    const PARAM_COUNT: usize = Q::PARAM_COUNT;

    #[inline]
    fn for_each_access_descriptor(f: impl FnMut(ComponentTypeId, &'static str, bool)) {
        Q::for_each_access_descriptor(f);
    }

    #[inline]
    fn validate_shape() -> Result<(), StateError> {
        Q::validate_shape()
    }

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<EntityId>() + Q::total_row_width()
    }

    #[inline]
    fn segment_end_with_cursor(buffers: &Self::ArchetypeBuffers, row: usize, cursor: &mut Self::Cursor) -> usize {
        Q::segment_end_with_cursor(buffers, row, cursor)
    }

    #[inline]
    unsafe fn prepare_iter<'view>(
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::IterState<'view> {
        unsafe { Q::prepare_iter(buffers, range, cursor) }
    }

    #[inline]
    unsafe fn next_item<'view>(
        state: &mut Self::IterState<'view>,
        entity: EntityId,
    ) -> Self::Item<'view> {
        unsafe { Q::next_item(state, entity) }
    }

    #[inline]
    unsafe fn skip_items<'view>(
        state: &mut Self::IterState<'view>,
        count: usize,
    ) {
        unsafe { Q::skip_items(state, count) }
    }

    #[inline]
    unsafe fn chunk<'view>(
        entities: &'view [EntityId],
        buffers: &Self::ArchetypeBuffers,
        range: Range<usize>,
        cursor: &mut Self::Cursor,
    ) -> Self::Chunk<'view> {
        unsafe { Q::chunk(entities, buffers, range, cursor) }
    }
}

impl<'a, Q: QueryMutData<'a>> QueryMutData<'a> for (EntityId, Q) {
    type Item = Q::Item;
    type Chunk = Q::Chunk;

    fn validate_query() {
        Q::validate_query();
    }
}

impl<'a, Q: QueryMutData<'a>> sealed::FetchMut<'a> for (EntityId, Q) {
    type State = Q::State;
    #[inline]
    fn prepare(
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<Self::State> {
        Q::prepare(columns, range)
    }

    #[inline(always)]
    fn fast_prepare_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<(Self::State, &'a [EntityId])> {
        Q::fast_prepare_archetype(archetype)
    }

    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<(Self::State, &'a [EntityId])> {
        Q::fast_prepare_bound(archetype, col_slots)
    }

    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryMutData<'a>>::Item {
        Q::next(state, entity)
    }

    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        Q::skip_rows(state, count);
    }

    #[inline]
    fn chunk(
        entities: &'a [EntityId],
        columns: &[(ComponentTypeId, ColumnDescriptor)],
        range: Range<usize>,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        Q::chunk(entities, columns, range)
    }

    #[inline(always)]
    fn fast_chunk_archetype(
        archetype: &'a mut Archetype,
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        Q::fast_chunk_archetype(archetype)
    }

    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a mut Archetype,
        col_slots: &[u16],
    ) -> Option<<Self as QueryMutData<'a>>::Chunk> {
        Q::fast_chunk_bound(archetype, col_slots)
    }

    #[inline(always)]
    fn mark_written_columns(
        archetype: &mut Archetype,
        masked: Option<&crate::storage::archetype::RowBitMask>,
        epoch: u64,
    ) {
        Q::mark_written_columns(archetype, masked, epoch);
    }

    #[inline]
    fn access_descriptors() -> smallvec::SmallVec<[(ComponentTypeId, &'static str, bool); 16]> {
        Q::access_descriptors()
    }

    #[inline]
    fn required_mask() -> u16 {
        Q::required_mask()
    }

    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<EntityId>() + Q::total_row_width()
    }
}

impl<'a, Q: QueryData<'a>> QueryData<'a> for (EntityId, Q) {
    type Item = Q::Item;
    type Chunk = Q::Chunk;
}

impl<'a, Q: QueryData<'a>> sealed::Fetch<'a> for (EntityId, Q) {
    type State = Q::State;
    #[inline]
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        Q::prepare(archetype, start)
    }
    #[inline(always)]
    fn fast_prepare_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(Self::State, usize)> {
        Q::fast_prepare_bound(archetype, col_slots, start)
    }
    #[inline]
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        Q::next(state, entity)
    }
    #[inline]
    fn skip_rows(state: &mut Self::State, count: usize) {
        Q::skip_rows(state, count);
    }
    #[inline]
    fn chunk(archetype: &'a Archetype, start: usize) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        Q::chunk(archetype, start)
    }
    #[inline(always)]
    fn fast_chunk_bound(
        archetype: &'a Archetype,
        col_slots: &[u16],
        start: usize,
    ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        Q::fast_chunk_bound(archetype, col_slots, start)
    }
    #[inline]
    fn total_row_width() -> usize {
        std::mem::size_of::<EntityId>() + Q::total_row_width()
    }
}

/// 64 行掩码字对齐的默认 CPU 并行子块大小（向后兼容回退常量：4096 行）。
pub const PAR_SUBCHUNK_ROWS: usize = 4096;

pub use crate::storage::primitives::{
    for_each_contiguous_mask_run, mask_has_any_in_range, SubchunkPlanner,
};

#[inline]
fn resolve_archetype_buffers_ro<Q: QueryMutShape>(
    arch: &Archetype,
) -> Option<Q::ArchetypeBuffers> {
    let len = arch.len();
    unsafe {
        Q::resolve_archetype_buffers(
            &|cid| {
                arch.columns()
                    .get(&cid)
                    .map(|col| ColumnDescriptor::from_column_ref(&**col))
            },
            len,
        )
    }
}

#[inline]
fn resolve_archetype_buffers_mut<Q: QueryMutShape>(
    arch: &mut Archetype,
) -> Option<Q::ArchetypeBuffers> {
    let len = arch.len();
    let wanted = Q::access_descriptors();
    let mut has_write = false;
    for &(_, _, is_write) in &wanted {
        if is_write {
            has_write = true;
            break;
        }
    }
    if !has_write {
        return resolve_archetype_buffers_ro::<Q>(&*arch);
    }
    // split_for_query_mut 内部已推进 columns_version += 1（用于使 PairTargetIndex 等按需缓存失效）
    let (_, _, _, _, cols) = arch.split_for_query_mut();
    let mut descs =
        smallvec::SmallVec::<[(ComponentTypeId, ColumnDescriptor); 8]>::with_capacity(wanted.len());
    for (cid, _, is_write) in wanted {
        let desc = if is_write {
            let col = cols.get_mut(&cid)?;
            ColumnDescriptor::from_column(&mut **col)
        } else {
            let col = cols.get(&cid)?;
            ColumnDescriptor::from_column_ref(&**col)
        };
        if desc.len != len {
            return None;
        }
        descs.push((cid, desc));
    }
    unsafe {
        Q::resolve_archetype_buffers(
            &|cid| {
                descs
                    .iter()
                    .find(|(id, _)| *id == cid)
                    .map(|(_, d)| d.clone())
            },
            len,
        )
    }
}

pub(crate) fn for_each_matched_chunk_ro<'store, Q, F>(
    store: &'store Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    mut f: impl FnMut(Q::Chunk<'store>),
) where
    Q: QueryMutShape,
    F: QueryFilter,
    for<'a> Q::View<'a>: QueryData<'a, Chunk = Q::Chunk<'a>>,
{
    for binding in matched_bindings {
        let Some(arch) = store.archetypes.get(binding.archetype_id.0 as usize) else {
            continue;
        };
        if arch.is_empty() {
            continue;
        }
        if F::IS_ROW_FILTERED {
            let Some(mask) = filter.collect_matched_row_mask(arch) else {
                continue;
            };
            if mask.is_all_zero() {
                continue;
            }
            let Some(buffers) = resolve_archetype_buffers_ro::<Q>(arch) else {
                continue;
            };
            let entities = arch.entities();
            let mut cursor = Q::Cursor::default();
            for_each_contiguous_mask_run(&mask, 0, arch.len(), |run| {
                let mut row = run.start;
                while row < run.end {
                    let seg_end =
                        Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(run.end);
                    if seg_end <= row {
                        break;
                    }
                    let chunk = unsafe {
                        Q::chunk(
                            &entities[row..seg_end],
                            &buffers,
                            row..seg_end,
                            &mut cursor,
                        )
                    };
                    f(chunk);
                    row = seg_end;
                }
            });
            continue;
        }
        let mut row = 0;
        while row < arch.len() {
            let Some((chunk, end)) = <Q::View<'store> as sealed::Fetch<'store>>::fast_chunk_bound(
                arch,
                &binding.flat_col_indices,
                row,
            ) else {
                break;
            };
            f(chunk);
            row = end;
        }
    }
}

pub(crate) fn for_each_matched_chunk_mut<'store, Q, F>(
    store: &'store mut Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    mut f: impl FnMut(Q::Chunk<'store>),
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    let write_epoch = store.epoch();
    let archetypes_ptr = store.archetypes.as_mut_ptr();
    let archetypes_len = store.archetypes.len();
    for binding in matched_bindings {
        let idx = binding.archetype_id.0 as usize;
        if idx >= archetypes_len {
            continue;
        }
        // SAFETY: `matched_bindings` contains distinct ArchetypeIds; we borrow each matched
        // archetype exclusively for the duration of its chunk callback.
        let arch = unsafe { &mut *archetypes_ptr.add(idx) };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter.collect_matched_row_mask(&*arch)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        <Q::View<'store> as sealed::FetchMut<'store>>::mark_written_columns(
            arch,
            filtered_mask.as_ref(),
            write_epoch,
        );
        if let Some(mask) = &filtered_mask {
            let arch_len = arch.len();
            let entities = unsafe { std::slice::from_raw_parts(arch.entities().as_ptr(), arch_len) };
            let Some(buffers) = resolve_archetype_buffers_mut::<Q>(arch) else {
                continue;
            };
            let mut cursor = Q::Cursor::default();
            for_each_contiguous_mask_run(mask, 0, arch_len, |run| {
                let mut row = run.start;
                while row < run.end {
                    let seg_end =
                        Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(run.end);
                    if seg_end <= row {
                        break;
                    }
                    let chunk = unsafe {
                        Q::chunk(
                            &entities[row..seg_end],
                            &buffers,
                            row..seg_end,
                            &mut cursor,
                        )
                    };
                    f(chunk);
                    row = seg_end;
                }
            });
            continue;
        }
        if !arch.may_be_segmented_or_unaligned()
            && arch.flat_columns().len() == arch.component_types_slice().len()
        {
            if let Some(chunk) = <Q::View<'store> as sealed::FetchMut<'store>>::fast_chunk_bound(
                arch,
                &binding.flat_col_indices,
            ) {
                f(chunk);
            }
            continue;
        }
        let (entities, _, _, _, cols) = arch.split_for_query_mut();
        let wanted = <Q::View<'store> as sealed::FetchMut<'store>>::access_descriptors();
        let req_mask = <Q::View<'store> as sealed::FetchMut<'store>>::required_mask();
        let mut columns =
            smallvec::SmallVec::<[(ComponentTypeId, ColumnDescriptor); 8]>::with_capacity(
                wanted.len(),
            );
        let mut ok = true;
        for (i, (id, _, is_write)) in wanted.into_iter().enumerate() {
            let is_req = (req_mask & (1u16 << i)) != 0;
            if let Some(col) = cols.get_mut(&id) {
                let desc = if is_write {
                    ColumnDescriptor::from_column(&mut **col)
                } else {
                    ColumnDescriptor::from_column_ref(&**col)
                };
                if desc.len != entities.len() {
                    ok = false;
                    break;
                }
                columns.push((id, desc));
            } else if is_req {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut row = 0;
        while row < entities.len() {
            let end = columns
                .iter()
                .fold(entities.len(), |e, (_, col)| e.min(col.locate(row).2));
            if end <= row {
                break;
            }
            let range = row..end;
            row = end;
            if let Some(chunk) = <Q::View<'store> as sealed::FetchMut<'store>>::chunk(
                &entities[range.clone()],
                &columns,
                range,
            ) {
                f(chunk);
            }
        }
    }
}

pub(crate) fn for_each_matched_ro<'store, Q, F>(
    store: &'store Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    mut f: impl FnMut(Q::Item<'store>),
) where
    Q: QueryMutShape,
    F: QueryFilter,
    for<'a> Q::View<'a>: QueryData<'a, Item = Q::Item<'a>>,
{
    for binding in matched_bindings {
        let Some(arch) = store.archetypes.get(binding.archetype_id.0 as usize) else {
            continue;
        };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter.collect_matched_row_mask(arch)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        let entities = arch.entities();
        let mut row = 0;
        while row < arch.len() {
            let Some((mut state, end)) =
                <Q::View<'store> as sealed::Fetch<'store>>::fast_prepare_bound(
                    arch,
                    &binding.flat_col_indices,
                    row,
                )
            else {
                break;
            };
            if let Some(mask) = &filtered_mask {
                let mut cursor_row = row;
                let start_word = row >> 6;
                let end_word = (end + 63) >> 6;
                let words = mask.words();
                for w_idx in start_word..end_word.min(words.len()) {
                    let mut w = words[w_idx];
                    if w_idx == start_word && (row & 63) != 0 {
                        w &= !((1u64 << (row & 63)) - 1);
                    }
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let target_row = w_idx * 64 + bit;
                        if target_row >= end {
                            break;
                        }
                        if target_row > cursor_row {
                            <Q::View<'store> as sealed::Fetch<'store>>::skip_rows(
                                &mut state,
                                target_row - cursor_row,
                            );
                        }
                        let item = <Q::View<'store> as sealed::Fetch<'store>>::next(
                            &mut state,
                            entities[target_row],
                        );
                        cursor_row = target_row + 1;
                        f(item);
                        w &= w - 1;
                    }
                }
            } else {
                for &entity in &entities[row..end] {
                    let item =
                        <Q::View<'store> as sealed::Fetch<'store>>::next(&mut state, entity);
                    f(item);
                }
            }
            row = end;
        }
    }
}

pub(crate) fn for_each_matched_mut<'store, Q, F>(
    store: &'store mut Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    mut f: impl FnMut(Q::Item<'store>),
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    let write_epoch = store.epoch();
    let archetypes_ptr = store.archetypes.as_mut_ptr();
    let archetypes_len = store.archetypes.len();
    for binding in matched_bindings {
        let idx = binding.archetype_id.0 as usize;
        if idx >= archetypes_len {
            continue;
        }
        // SAFETY: `matched_bindings` contains distinct ArchetypeIds; we borrow each matched
        // archetype exclusively for the duration of its row callbacks.
        let arch = unsafe { &mut *archetypes_ptr.add(idx) };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter.collect_matched_row_mask(&*arch)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        <Q::View<'store> as sealed::FetchMut<'store>>::mark_written_columns(
            arch,
            filtered_mask.as_ref(),
            write_epoch,
        );
        if !arch.may_be_segmented_or_unaligned()
            && arch.flat_columns().len() == arch.component_types_slice().len()
        {
            if let Some((mut state, entities)) =
                <Q::View<'store> as sealed::FetchMut<'store>>::fast_prepare_bound(
                    arch,
                    &binding.flat_col_indices,
                )
            {
                if let Some(mask) = &filtered_mask {
                    let mut cursor_row = 0usize;
                    mask.for_each_set_row(|target_row| {
                        if target_row < entities.len() {
                            if target_row > cursor_row {
                                <Q::View<'store> as sealed::FetchMut<'store>>::skip_rows(
                                    &mut state,
                                    target_row - cursor_row,
                                );
                            }
                            let item = <Q::View<'store> as sealed::FetchMut<'store>>::next(
                                &mut state,
                                entities[target_row],
                            );
                            cursor_row = target_row + 1;
                            f(item);
                        }
                    });
                } else {
                    for &entity in entities {
                        let item = <Q::View<'store> as sealed::FetchMut<'store>>::next(
                            &mut state,
                            entity,
                        );
                        f(item);
                    }
                }
            }
            continue;
        }
        let (entities, _, _, _, cols) = arch.split_for_query_mut();
        let wanted = <Q::View<'store> as sealed::FetchMut<'store>>::access_descriptors();
        let req_mask = <Q::View<'store> as sealed::FetchMut<'store>>::required_mask();
        let mut columns =
            smallvec::SmallVec::<[(ComponentTypeId, ColumnDescriptor); 8]>::with_capacity(
                wanted.len(),
            );
        let mut ok = true;
        for (i, (id, _, is_write)) in wanted.into_iter().enumerate() {
            let is_req = (req_mask & (1u16 << i)) != 0;
            if let Some(col) = cols.get_mut(&id) {
                let desc = if is_write {
                    ColumnDescriptor::from_column(&mut **col)
                } else {
                    ColumnDescriptor::from_column_ref(&**col)
                };
                if desc.len != entities.len() {
                    ok = false;
                    break;
                }
                columns.push((id, desc));
            } else if is_req {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut row = 0;
        while row < entities.len() {
            let end = columns
                .iter()
                .fold(entities.len(), |e, (_, col)| e.min(col.locate(row).2));
            if end <= row {
                break;
            }
            let range = row..end;
            if let Some(mut state) =
                <Q::View<'store> as sealed::FetchMut<'store>>::prepare(&columns, range.clone())
            {
                if let Some(mask) = &filtered_mask {
                    let mut cursor_row = row;
                    let start_word = row >> 6;
                    let end_word = (end + 63) >> 6;
                    let words = mask.words();
                    for w_idx in start_word..end_word.min(words.len()) {
                        let mut w = words[w_idx];
                        if w_idx == start_word && (row & 63) != 0 {
                            w &= !((1u64 << (row & 63)) - 1);
                        }
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            let target_row = w_idx * 64 + bit;
                            if target_row >= end {
                                break;
                            }
                            if target_row > cursor_row {
                                <Q::View<'store> as sealed::FetchMut<'store>>::skip_rows(
                                    &mut state,
                                    target_row - cursor_row,
                                );
                            }
                            let item = <Q::View<'store> as sealed::FetchMut<'store>>::next(
                                &mut state,
                                entities[target_row],
                            );
                            cursor_row = target_row + 1;
                            f(item);
                            w &= w - 1;
                        }
                    }
                } else {
                    for &entity in &entities[range] {
                        let item = <Q::View<'store> as sealed::FetchMut<'store>>::next(
                            &mut state,
                            entity,
                        );
                        f(item);
                    }
                }
            }
            row = end;
        }
    }
}

struct ParSubchunkTask<'store, B> {
    entities: &'store [EntityId],
    buffers: B,
    range: Range<usize>,
    filtered_mask: Option<std::sync::Arc<crate::storage::archetype::RowBitMask>>,
}

#[inline]
fn exec_subchunk_row_task<'store, Q>(
    task: ParSubchunkTask<'store, Q::ArchetypeBuffers>,
    f: &impl Fn(Q::Item<'store>),
) where
    Q: QueryMutShape,
{
    let mut cursor = Q::Cursor::default();
    let row = task.range.start;
    let end = task.range.end;
    let mut state: Q::IterState<'store> =
        unsafe { Q::prepare_iter(&task.buffers, row..end, &mut cursor) };
    if let Some(mask) = &task.filtered_mask {
        let mut cursor_row = row;
        let start_word = row >> 6;
        let end_word = (end + 63) >> 6;
        let words = mask.words();
        for w_idx in start_word..end_word.min(words.len()) {
            let mut w = words[w_idx];
            if w_idx == start_word && (row & 63) != 0 {
                w &= !((1u64 << (row & 63)) - 1);
            }
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let target_row = w_idx * 64 + bit;
                if target_row >= end {
                    break;
                }
                if target_row > cursor_row {
                    unsafe { Q::skip_items(&mut state, target_row - cursor_row) };
                }
                let item =
                    unsafe { Q::next_item(&mut state, task.entities[target_row]) };
                cursor_row = target_row + 1;
                f(item);
                w &= w - 1;
            }
        }
    } else {
        for &entity in &task.entities[row..end] {
            let item = unsafe { Q::next_item(&mut state, entity) };
            f(item);
        }
    }
}

#[inline]
fn exec_subchunk_slice_task<'store, Q>(
    task: ParSubchunkTask<'store, Q::ArchetypeBuffers>,
    f: &impl Fn(Q::Chunk<'store>),
) where
    Q: QueryMutShape,
{
    let mut cursor = Q::Cursor::default();
    if let Some(mask) = &task.filtered_mask {
        for_each_contiguous_mask_run(mask, task.range.start, task.range.end, |run| {
            let chunk = unsafe {
                Q::chunk(
                    &task.entities[run.clone()],
                    &task.buffers,
                    run,
                    &mut cursor,
                )
            };
            f(chunk);
        });
    } else {
        let chunk = unsafe {
            Q::chunk(
                &task.entities[task.range.clone()],
                &task.buffers,
                task.range,
                &mut cursor,
            )
        };
        f(chunk);
    }
}

/// 基于 Rayon 的按 4096 行（64 个掩码字）子块并行只读行遍历。
///
/// - 在主线程预先计算每个匹配原型的 `RowBitMask` 并对全 0 子块执行 $O(1)$ 剪枝；
/// - 若有效子块总数 $\le 1$，直接在当前线程内联执行，避免线程池唤醒开销；
/// - 若有效子块数 $> 1$，通过 `rayon::into_par_iter` 在多核间无锁并行分发。
pub(crate) fn par_for_each_matched_ro<'store, Q, F>(
    store: &'store Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    f: impl Fn(Q::Item<'store>) + Send + Sync,
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    use rayon::prelude::*;

    let subchunk_rows = SubchunkPlanner::default().subchunk_rows(Q::total_row_width());
    let mut tasks: Vec<ParSubchunkTask<'store, Q::ArchetypeBuffers>> = Vec::new();
    for binding in matched_bindings {
        let Some(arch) = store.archetypes.get(binding.archetype_id.0 as usize) else {
            continue;
        };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter
                .collect_matched_row_mask(arch)
                .map(std::sync::Arc::new)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        let Some(buffers) = resolve_archetype_buffers_ro::<Q>(arch) else {
            continue;
        };
        let entities = arch.entities();
        let len = arch.len();
        let mut cursor = Q::Cursor::default();
        let mut row = 0;
        while row < len {
            let seg_end = Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(len);
            if seg_end <= row {
                break;
            }
            let sub_end = (row + subchunk_rows).min(seg_end);
            let has_match = filtered_mask
                .as_ref()
                .map_or(true, |m| mask_has_any_in_range(m, row, sub_end));
            if has_match {
                tasks.push(ParSubchunkTask {
                    entities,
                    buffers: buffers.clone(),
                    range: row..sub_end,
                    filtered_mask: filtered_mask.clone(),
                });
            }
            row = sub_end;
        }
    }
    match tasks.len() {
        0 => {}
        1 => {
            if let Some(task) = tasks.pop() {
                exec_subchunk_row_task::<Q>(task, &f);
            }
        }
        _ => {
            tasks
                .into_par_iter()
                .for_each(|task| exec_subchunk_row_task::<Q>(task, &f));
        }
    }
}

/// 基于 Rayon 的按 4096 行（64 个掩码字）子块并行可变行遍历。
///
/// - 在主线程独占完成 COW 解耦（仅针对写权限列触发，只读列保持零拷贝）、64 行块纪元标记（`mark_written_columns`）与 `RowBitMask` 求交；
/// - 将每张原型表切分为互不重叠的 4096 行（64 字对齐）`Range<usize>` 子块并在多核间并行执行。
pub(crate) fn par_for_each_matched_mut<'store, Q, F>(
    store: &'store mut Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    f: impl Fn(Q::Item<'store>) + Send + Sync,
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    let write_epoch = store.epoch();
    prepare_par_matched_mut_tasks::<Q, F>(store, matched_bindings, filter, write_epoch).execute(&f);
}

/// 主线程预备完毕、可脱离 `Store` 独立执行的可变子块任务集合。
///
/// # 两阶段执行契约（调度器并行 Stage 的健全性基础）
///
/// - **准备阶段**（[`prepare_par_matched_mut_tasks`]，主线程独占 `&mut Store`）：完成写列 COW 解耦、
///   64 行块纪元标记与行级过滤掩码求值，并把每张原型表切分为互不重叠的子块；
/// - **执行阶段**（[`execute`](Self::execute) / [`execute_with_state`](Self::execute_with_state)）：
///   任务仅持有列缓冲区与实体切片的物理指针，**不持有也不访问任何 `Store` / `Archetype` 引用**，
///   因此多个读写集互不相交的系统可在同一 Stage 内安全并发执行。
pub(crate) struct PreparedParTasks<'store, Q: QueryMutShape> {
    tasks: Vec<ParSubchunkTask<'store, Q::ArchetypeBuffers>>,
}

impl<'store, Q: QueryMutShape> PreparedParTasks<'store, Q> {
    /// 子块任务数
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.tasks.len()
    }

    /// 并行执行全部子块（单任务时内联执行，避免线程池唤醒开销）
    pub(crate) fn execute(self, f: &(impl Fn(Q::Item<'store>) + Send + Sync)) {
        use rayon::prelude::*;
        let mut tasks = self.tasks;
        match tasks.len() {
            0 => {}
            1 => {
                if let Some(task) = tasks.pop() {
                    exec_subchunk_row_task::<Q>(task, f);
                }
            }
            _ => {
                tasks
                    .into_par_iter()
                    .for_each(|task| exec_subchunk_row_task::<Q>(task, f));
            }
        }
    }

    /// 并行执行全部子块，每个子块持有独立的 `S` 状态（如命令缓冲），按**任务序**返回。
    ///
    /// 任务序 = 匹配原型序 × 行序，与线程调度无关，从而保证下游合并结果确定。
    pub(crate) fn execute_with_state<S: Default + Send>(
        self,
        f: &(impl Fn(Q::Item<'store>, &mut S) + Send + Sync),
    ) -> Vec<S> {
        use rayon::prelude::*;
        let run_one = |task: ParSubchunkTask<'store, Q::ArchetypeBuffers>| -> S {
            let state = std::cell::RefCell::new(S::default());
            exec_subchunk_row_task::<Q>(task, &|item| f(item, &mut state.borrow_mut()));
            state.into_inner()
        };
        let tasks = self.tasks;
        if tasks.len() <= 1 {
            tasks.into_iter().map(run_one).collect()
        } else {
            tasks.into_par_iter().map(run_one).collect()
        }
    }

    /// 并行执行全部连续切片子块，每个子块持有独立的 `S` 状态（如命令缓冲），按**任务序**返回。
    ///
    /// 若存在行级过滤器，每个子块只把通过过滤的连续行区间（Contiguous Mask Runs）以切片形式喂给闭包，
    /// 适合 SIMD / 向量化批处理系统（[`ParChunkQuerySystem`](crate::schedule::ParChunkQuerySystem)）。
    pub(crate) fn execute_chunks_with_state<S: Default + Send>(
        self,
        f: &(impl Fn(Q::Chunk<'store>, &mut S) + Send + Sync),
    ) -> Vec<S> {
        use rayon::prelude::*;
        let run_one = |task: ParSubchunkTask<'store, Q::ArchetypeBuffers>| -> S {
            let state = std::cell::RefCell::new(S::default());
            exec_subchunk_slice_task::<Q>(task, &|chunk| f(chunk, &mut state.borrow_mut()));
            state.into_inner()
        };
        let tasks = self.tasks;
        if tasks.len() <= 1 {
            tasks.into_iter().map(run_one).collect()
        } else {
            tasks.into_par_iter().map(run_one).collect()
        }
    }
}

/// 两阶段并行可变遍历的**准备阶段**（必须在持有 `&mut Store` 的主线程上调用）。
///
/// `write_epoch` 为本次写入要打上的 64 行块纪元；调用方负责事先推进 `Store` 纪元。
pub(crate) fn prepare_par_matched_mut_tasks<'store, Q, F>(
    store: &'store mut Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    write_epoch: u64,
) -> PreparedParTasks<'store, Q>
where
    Q: QueryMutShape,
    F: QueryFilter,
{
    let subchunk_rows = SubchunkPlanner::default().subchunk_rows(Q::total_row_width());
    let archetypes_ptr = store.archetypes.as_mut_ptr();
    let archetypes_len = store.archetypes.len();
    let mut tasks: Vec<ParSubchunkTask<'store, Q::ArchetypeBuffers>> = Vec::new();

    for binding in matched_bindings {
        let idx = binding.archetype_id.0 as usize;
        if idx >= archetypes_len {
            continue;
        }
        let arch = unsafe { &mut *archetypes_ptr.add(idx) };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter
                .collect_matched_row_mask(&*arch)
                .map(std::sync::Arc::new)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        <Q::View<'store> as sealed::FetchMut<'store>>::mark_written_columns(
            arch,
            filtered_mask.as_deref(),
            write_epoch,
        );
        let len = arch.len();
        let entities = unsafe { std::slice::from_raw_parts(arch.entities().as_ptr(), len) };
        let Some(buffers) = resolve_archetype_buffers_mut::<Q>(arch) else {
            continue;
        };
        let mut cursor = Q::Cursor::default();
        let mut row = 0;
        while row < len {
            let seg_end = Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(len);
            if seg_end <= row {
                break;
            }
            let sub_end = (row + subchunk_rows).min(seg_end);
            let has_match = filtered_mask
                .as_ref()
                .map_or(true, |m| mask_has_any_in_range(m, row, sub_end));
            if has_match {
                tasks.push(ParSubchunkTask {
                    entities,
                    buffers: buffers.clone(),
                    range: row..sub_end,
                    filtered_mask: filtered_mask.clone(),
                });
            }
            row = sub_end;
        }
    }
    PreparedParTasks { tasks }
}

/// 基于 Rayon 的按 4096 行子块并行只读连续切片遍历（支持 `ChangedSince` 与行级掩码连续段切分）。
pub(crate) fn par_for_each_matched_chunk_ro<'store, Q, F>(
    store: &'store Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    f: impl Fn(Q::Chunk<'store>) + Send + Sync,
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    use rayon::prelude::*;

    let subchunk_rows = SubchunkPlanner::default().subchunk_rows(Q::total_row_width());
    let mut tasks: Vec<ParSubchunkTask<'store, Q::ArchetypeBuffers>> = Vec::new();
    for binding in matched_bindings {
        let Some(arch) = store.archetypes.get(binding.archetype_id.0 as usize) else {
            continue;
        };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter.collect_matched_row_mask(arch)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        let Some(buffers) = resolve_archetype_buffers_ro::<Q>(arch) else {
            continue;
        };
        let entities = arch.entities();
        let len = arch.len();
        let mut cursor = Q::Cursor::default();
        if let Some(mask) = &filtered_mask {
            for_each_contiguous_mask_run(mask, 0, len, |run| {
                let mut row = run.start;
                while row < run.end {
                    let seg_end =
                        Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(run.end);
                    if seg_end <= row {
                        break;
                    }
                    let sub_end = (row + subchunk_rows).min(seg_end);
                    tasks.push(ParSubchunkTask {
                        entities,
                        buffers: buffers.clone(),
                        range: row..sub_end,
                        filtered_mask: None,
                    });
                    row = sub_end;
                }
            });
        } else {
            let mut row = 0;
            while row < len {
                let seg_end = Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(len);
                if seg_end <= row {
                    break;
                }
                let sub_end = (row + subchunk_rows).min(seg_end);
                tasks.push(ParSubchunkTask {
                    entities,
                    buffers: buffers.clone(),
                    range: row..sub_end,
                    filtered_mask: None,
                });
                row = sub_end;
            }
        }
    }
    match tasks.len() {
        0 => {}
        1 => {
            if let Some(task) = tasks.pop() {
                exec_subchunk_slice_task::<Q>(task, &f);
            }
        }
        _ => {
            tasks
                .into_par_iter()
                .for_each(|task| exec_subchunk_slice_task::<Q>(task, &f));
        }
    }
}

/// 基于 Rayon 的按 4096 行子块并行可变连续切片遍历（支持 `ChangedSince` 与行级掩码连续段切分）。
pub(crate) fn par_for_each_matched_chunk_mut<'store, Q, F>(
    store: &'store mut Store,
    matched_bindings: &[crate::storage::query::handle::MatchedTableBinding],
    filter: &F,
    f: impl Fn(Q::Chunk<'store>) + Send + Sync,
) where
    Q: QueryMutShape,
    F: QueryFilter,
{
    use rayon::prelude::*;

    let subchunk_rows = SubchunkPlanner::default().subchunk_rows(Q::total_row_width());
    let write_epoch = store.epoch();
    let archetypes_ptr = store.archetypes.as_mut_ptr();
    let archetypes_len = store.archetypes.len();
    let mut tasks: Vec<ParSubchunkTask<'store, Q::ArchetypeBuffers>> = Vec::new();

    for binding in matched_bindings {
        let idx = binding.archetype_id.0 as usize;
        if idx >= archetypes_len {
            continue;
        }
        let arch = unsafe { &mut *archetypes_ptr.add(idx) };
        if arch.is_empty() {
            continue;
        }
        let filtered_mask = if F::IS_ROW_FILTERED {
            filter.collect_matched_row_mask(&*arch)
        } else {
            None
        };
        if filtered_mask.as_ref().is_some_and(|m| m.is_all_zero()) {
            continue;
        }
        <Q::View<'store> as sealed::FetchMut<'store>>::mark_written_columns(
            arch,
            filtered_mask.as_ref(),
            write_epoch,
        );
        let len = arch.len();
        let entities = unsafe { std::slice::from_raw_parts(arch.entities().as_ptr(), len) };
        let Some(buffers) = resolve_archetype_buffers_mut::<Q>(arch) else {
            continue;
        };
        let mut cursor = Q::Cursor::default();
        if let Some(mask) = &filtered_mask {
            for_each_contiguous_mask_run(mask, 0, len, |run| {
                let mut row = run.start;
                while row < run.end {
                    let seg_end =
                        Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(run.end);
                    if seg_end <= row {
                        break;
                    }
                    let sub_end = (row + subchunk_rows).min(seg_end);
                    tasks.push(ParSubchunkTask {
                        entities,
                        buffers: buffers.clone(),
                        range: row..sub_end,
                        filtered_mask: None,
                    });
                    row = sub_end;
                }
            });
        } else {
            let mut row = 0;
            while row < len {
                let seg_end = Q::segment_end_with_cursor(&buffers, row, &mut cursor).min(len);
                if seg_end <= row {
                    break;
                }
                let sub_end = (row + subchunk_rows).min(seg_end);
                tasks.push(ParSubchunkTask {
                    entities,
                    buffers: buffers.clone(),
                    range: row..sub_end,
                    filtered_mask: None,
                });
                row = sub_end;
            }
        }
    }
    match tasks.len() {
        0 => {}
        1 => {
            if let Some(task) = tasks.pop() {
                exec_subchunk_slice_task::<Q>(task, &f);
            }
        }
        _ => {
            tasks
                .into_par_iter()
                .for_each(|task| exec_subchunk_slice_task::<Q>(task, &f));
        }
    }
}



