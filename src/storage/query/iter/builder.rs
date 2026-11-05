use super::*;

#[must_use = "a query does no work until iterated"]
pub struct QueryMutBuilder<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    archetypes: &'a mut [Archetype],
    row_upper_bound: usize,
    write_epoch: u64,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::fmt::Debug for QueryMutBuilder<'a, Q, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryMutBuilder")
            .field("row_upper_bound", &self.row_upper_bound)
            .finish()
    }
}

impl<'a, Q: QueryMutData<'a>> QueryMutBuilder<'a, Q> {
    pub(crate) fn new(store: &'a mut crate::storage::store::Store) -> Self {
        let row_upper_bound = store.len();
        let write_epoch = store.epoch();
        Self {
            archetypes: &mut store.archetypes,
            row_upper_bound,
            write_epoch,
            filter: All,
            marker: PhantomData,
        }
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> QueryMutBuilder<'a, Q, F> {
    /// Require component T without examining or borrowing its values.
    #[inline]
    pub fn with<T: 'static>(self) -> QueryMutBuilder<'a, Q, With<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: With {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing component T.
    #[inline]
    pub fn without<T: 'static>(self) -> QueryMutBuilder<'a, Q, Without<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: Without {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Require relationship pair `(R, target)` on the entity (accelerated via per-table `PairTargetIndex`).
    #[inline]
    pub fn with_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        self,
        target: EntityId,
    ) -> QueryMutBuilder<'a, Q, WithPairTarget<R, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithPairTarget::new(self.filter, target),
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing relationship pair `Pair<R>`.
    #[inline]
    pub fn without_pair<R: 'static>(
        self,
    ) -> QueryMutBuilder<'a, Q, Without<crate::al::Pair<R>, F>> {
        self.without::<crate::al::Pair<R>>()
    }

    /// Require zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn with_tag<const N: u8>(self) -> QueryMutBuilder<'a, Q, With<crate::al::Tag<N>, F>> {
        self.with::<crate::al::Tag<N>>()
    }

    /// Skip entire tables containing zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn without_tag<const N: u8>(
        self,
    ) -> QueryMutBuilder<'a, Q, Without<crate::al::Tag<N>, F>> {
        self.without::<crate::al::Tag<N>>()
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be set (`1`).
    #[inline]
    pub fn with_flag<const FLAG: u8>(self) -> QueryMutBuilder<'a, Q, WithFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be unset (`0`).
    #[inline]
    pub fn without_flag<const FLAG: u8>(self) -> QueryMutBuilder<'a, Q, WithoutFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithoutFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Filter to 64-row chunks of component `T` modified strictly after `since_epoch`.
    #[inline]
    pub fn changed_since<T: 'static>(
        self,
        since_epoch: u64,
    ) -> QueryMutBuilder<'a, Q, ChangedSince<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: ChangedSince::new(self.filter, since_epoch),
            marker: PhantomData,
        }
    }

    /// Add a row-level component predicate filter.
    #[inline]
    pub fn filter<C: ModelComponent + Serialize + DeserializeOwned>(
        self,
        predicate: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> QueryMutBuilder<'a, Q, BusinessFilter<F, C>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: BusinessFilter::new(self.filter, predicate),
            marker: PhantomData,
        }
    }

    /// Add a row-level component field equality filter.
    #[inline]
    pub fn filter_eq<C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>>(
        self,
        expected: Field::Value,
    ) -> QueryMutBuilder<'a, Q, FilterEq<F, C, Field>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: FilterEq::new(self.filter, expected),
            marker: PhantomData,
        }
    }

    /// Consume the builder into a lazy, forward-only row iterator.
    #[inline]
    pub fn iter_mut(self) -> QueryMutIter<'a, Q, F> {
        QueryMutIter {
            archetypes: self.archetypes.iter_mut(),
            entities: [].iter(),
            segment_start_row: 0,
            segment_offset: 0,
            filtered_mask: None,
            state: None,
            current: None,
            remaining_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: self.filter,
        }
    }

    /// Alias for `iter_mut()`.
    #[inline]
    pub fn iter(self) -> QueryMutIter<'a, Q, F> {
        self.iter_mut()
    }

    /// Consume the builder into an archetype-level contiguous slice iterator.
    #[inline]
    pub fn chunks_mut(self) -> QueryMutChunksIter<'a, Q, F> {
        QueryMutChunksIter {
            archetypes: self.archetypes.iter_mut(),
            current: None,
            write_epoch: self.write_epoch,
            filter: self.filter,
            marker: PhantomData,
        }
    }

    /// Alias for `chunks_mut()`.
    #[inline]
    pub fn chunks(self) -> QueryMutChunksIter<'a, Q, F> {
        self.chunks_mut()
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> IntoIterator for QueryMutBuilder<'a, Q, F> {
    type Item = Q::Item;
    type IntoIter = QueryMutIter<'a, Q, F>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Descriptors are captured before lending any reference, then only raw
/// disjoint ranges are derived. Values remain owned by the exclusively borrowed Store.
pub(crate) struct MutableTable<'a> {
    entities: &'a [EntityId],
    columns: smallvec::SmallVec<[(ComponentTypeId, ColumnDescriptor); 4]>,
    row: usize,
}
impl<'a> MutableTable<'a> {
    #[inline]
    fn new<Q: QueryMutData<'a>>(archetype: &'a mut Archetype) -> Option<Self> {
        let wanted = Q::access_descriptors();
        let req_mask = Q::required_mask();
        let (entities, comp_slice, flat_cols, may_be_segmented, cols) =
            archetype.split_for_query_mut();
        let mut columns = smallvec::SmallVec::with_capacity(wanted.len());
        if !may_be_segmented && flat_cols.len() == comp_slice.len() {
            for (idx, (id, _, _)) in wanted.into_iter().enumerate() {
                let is_req = (req_mask & (1u16 << idx)) != 0;
                if let Some(pos) = comp_slice.iter().position(|&cid| cid == id) {
                    let entry = unsafe { *flat_cols.get_unchecked(pos) };
                    let desc = ColumnDescriptor::from_column(unsafe { &mut *entry.col_ptr });
                    if desc.len != entities.len() {
                        return None;
                    }
                    columns.push((id, desc));
                } else if is_req {
                    return None;
                }
            }
            return Some(Self {
                entities,
                columns,
                row: 0,
            });
        }
        for (idx, (id, _, _)) in wanted.iter().enumerate() {
            let is_req = (req_mask & (1u16 << idx)) != 0;
            if is_req && !comp_slice.contains(id) {
                return None;
            }
        }
        for (idx, (id, _, _)) in wanted.into_iter().enumerate() {
            let is_req = (req_mask & (1u16 << idx)) != 0;
            if let Some(col) = cols.get_mut(&id) {
                let desc = ColumnDescriptor::from_column(&mut **col);
                if desc.len != entities.len() {
                    return None;
                }
                columns.push((id, desc));
            } else if is_req || comp_slice.contains(&id) {
                return None;
            }
        }
        Some(Self {
            entities,
            columns,
            row: 0,
        })
    }
    fn next_range(&mut self) -> Option<Range<usize>> {
        if self.row >= self.entities.len() { return None; }
        let end = self.columns.iter().fold(self.entities.len(), |end, (_, col)| end.min(col.locate(self.row).2));
        assert!(end > self.row);
        let range = self.row..end;
        self.row = end;
        Some(range)
    }
}

/// Forward-only row iterator over mutable component slices.
pub struct QueryMutIter<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::IterMut<'a, Archetype>,
    entities: std::slice::Iter<'a, EntityId>,
    segment_start_row: usize,
    segment_offset: usize,
    filtered_mask: Option<crate::storage::archetype::RowBitMask>,
    state: Option<Q::State>,
    current: Option<MutableTable<'a>>,
    remaining_upper_bound: usize,
    write_epoch: u64,
    filter: F,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> Iterator for QueryMutIter<'a, Q, F> {
    type Item = Q::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.entities.as_slice().is_empty() {
                if let Some(mask) = &self.filtered_mask {
                    let rem_len = self.entities.len();
                    let start_row = self.segment_start_row + self.segment_offset;
                    let seg_end = start_row + rem_len;
                    let mut cur = start_row;
                    while cur < seg_end {
                        let w_idx = cur >> 6;
                        let bit_idx = cur & 63;
                        let word = mask.words().get(w_idx).copied().unwrap_or(0) >> bit_idx;
                        if word != 0 {
                            cur += word.trailing_zeros() as usize;
                            break;
                        }
                        cur = (w_idx + 1) << 6;
                    }
                    let skip = cur.min(seg_end) - start_row;
                    if skip > 0 {
                        self.segment_offset += skip;
                        self.remaining_upper_bound -= skip;
                        if skip >= rem_len {
                            self.entities = [].iter();
                            continue;
                        }
                        self.entities = self.entities.as_slice()[skip..].iter();
                        Q::skip_rows(
                            self.state.as_mut().expect("prepared component slices"),
                            skip,
                        );
                    }
                }
                if let Some(&entity) = self.entities.next() {
                    self.segment_offset += 1;
                    self.remaining_upper_bound -= 1;
                    let item = Q::next(
                        self.state.as_mut().expect("prepared component slices"),
                        entity,
                    );
                    return Some(item);
                }
            }
            self.state = None;
            if let Some(table) = &mut self.current {
                if let Some(range) = table.next_range() {
                    self.segment_start_row = range.start;
                    self.segment_offset = 0;
                    self.state = Q::prepare(&table.columns, range.clone());
                    self.entities = table.entities[range].iter();
                    continue;
                }
            }
            self.current = None;
            self.filtered_mask = None;
            let archetype = self.archetypes.next()?;
            let arch_len = archetype.len();
            if arch_len != 0 && self.filter.matches_archetype(archetype) {
                if F::IS_ROW_FILTERED {
                    let mask = self.filter.collect_matched_row_mask(archetype);
                    if mask.as_ref().is_some_and(|m| m.is_all_zero()) {
                        self.remaining_upper_bound -= arch_len;
                        continue;
                    }
                    self.filtered_mask = mask;
                }
                Q::mark_written_columns(
                    archetype,
                    self.filtered_mask.as_ref(),
                    self.write_epoch,
                );
                if !archetype.may_be_segmented_or_unaligned()
                    && archetype.flat_columns().len() == archetype.component_types_slice().len()
                {
                    if let Some((state, entities)) = Q::fast_prepare_archetype(archetype) {
                        self.segment_start_row = 0;
                        self.segment_offset = 0;
                        self.state = Some(state);
                        self.entities = entities.iter();
                        continue;
                    }
                    self.remaining_upper_bound -= arch_len;
                    continue;
                }
                self.current = MutableTable::new::<Q>(archetype);
                if self.current.is_some() { continue; }
            }
            self.remaining_upper_bound -= arch_len;
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let lower = if self.filtered_mask.is_some() {
            0
        } else {
            self.entities.len()
        };
        (lower, Some(self.remaining_upper_bound))
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryMutIter<'a, Q, F> {}

/// Archetype-level iterator over mutable contiguous component slices.
pub struct QueryMutChunksIter<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::IterMut<'a, Archetype>,
    current: Option<MutableTable<'a>>,
    write_epoch: u64,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> Iterator for QueryMutChunksIter<'a, Q, F> {
    type Item = Q::Chunk;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(table) = &mut self.current {
                if let Some(range) = table.next_range() {
                    return Q::chunk(&table.entities[range.clone()], &table.columns, range);
                }
            }
            self.current = None;
            let archetype = self.archetypes.next()?;
            if !archetype.is_empty() && self.filter.matches_archetype(archetype) {
                Q::mark_written_columns(archetype, None, self.write_epoch);
                if !archetype.may_be_segmented_or_unaligned()
                    && archetype.flat_columns().len() == archetype.component_types_slice().len()
                {
                    if let Some(chunk) = Q::fast_chunk_archetype(archetype) {
                        return Some(chunk);
                    }
                    continue;
                }
                self.current = MutableTable::new::<Q>(archetype);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::iter::FusedIterator
    for QueryMutChunksIter<'a, Q, F>
{
}

impl Store {
    /// 导出匹配类型的连续列切片块迭代器
    #[inline]
    pub fn query_chunks<T: ModelComponent + Serialize + DeserializeOwned>(
        &self,
    ) -> QueryChunks<'_, T> {
        QueryChunks::new(&self.archetypes)
    }

    /// 开启只读元组集合查询构建器
    #[inline]
    pub fn query<'a, Q: QueryData<'a>>(
        &'a self,
    ) -> SelectBuilder<'a, Q> {
        SelectBuilder::new(self)
    }

    /// 开启排他连续批量直接写访问构建器
    #[inline]
    pub fn query_mut<'a, Q: QueryMutData<'a>>(
        &'a mut self,
    ) -> Result<QueryMutBuilder<'a, Q>, StateError> {
        self.check_poisoned()?;
        Q::validate_query();
        self.next_epoch()?;
        self.bump_epoch();
        Ok(QueryMutBuilder::new(self))
    }


}

