use super::*;

#[must_use = "a query does no work until iterated"]
pub struct SelectBuilder<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    archetypes: &'a [Archetype],
    row_upper_bound: usize,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::fmt::Debug for SelectBuilder<'a, Q, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectBuilder")
            .field("row_upper_bound", &self.row_upper_bound)
            .finish()
    }
}

impl<'a, Q: QueryData<'a>> SelectBuilder<'a, Q> {
    pub(crate) fn new(store: &'a crate::storage::store::Store) -> Self {
        Self {
            archetypes: &store.archetypes,
            row_upper_bound: store.len(),
            filter: All,
            marker: PhantomData,
        }
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> SelectBuilder<'a, Q, F> {
    /// Require component T without borrowing or examining its values.
    #[inline]
    pub fn with<T: 'static>(self) -> SelectBuilder<'a, Q, With<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: With {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing component T, before extracting any slices.
    #[inline]
    pub fn without<T: 'static>(self) -> SelectBuilder<'a, Q, Without<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
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
    ) -> SelectBuilder<'a, Q, WithPairTarget<R, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithPairTarget::new(self.filter, target),
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing relationship pair `Pair<R>`.
    #[inline]
    pub fn without_pair<R: 'static>(self) -> SelectBuilder<'a, Q, Without<crate::al::Pair<R>, F>> {
        self.without::<crate::al::Pair<R>>()
    }

    /// Require zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn with_tag<const N: u8>(self) -> SelectBuilder<'a, Q, With<crate::al::Tag<N>, F>> {
        self.with::<crate::al::Tag<N>>()
    }

    /// Skip entire tables containing zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn without_tag<const N: u8>(self) -> SelectBuilder<'a, Q, Without<crate::al::Tag<N>, F>> {
        self.without::<crate::al::Tag<N>>()
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be set (`1`).
    #[inline]
    pub fn with_flag<const FLAG: u8>(self) -> SelectBuilder<'a, Q, WithFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be unset (`0`).
    #[inline]
    pub fn without_flag<const FLAG: u8>(self) -> SelectBuilder<'a, Q, WithoutFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithoutFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Filter to 64-row chunks of component `T` modified strictly after `since_epoch`.
    #[inline]
    pub fn changed_since<T: 'static>(
        self,
        since_epoch: u64,
    ) -> SelectBuilder<'a, Q, ChangedSince<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: ChangedSince::new(self.filter, since_epoch),
            marker: PhantomData,
        }
    }

    /// Add a row-level component predicate filter.
    #[inline]
    pub fn filter<C: ModelComponent + Serialize + DeserializeOwned>(
        self,
        predicate: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> SelectBuilder<'a, Q, BusinessFilter<F, C>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: BusinessFilter::new(self.filter, predicate),
            marker: PhantomData,
        }
    }

    /// Add a row-level component field equality filter.
    #[inline]
    pub fn filter_eq<C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>>(
        self,
        expected: Field::Value,
    ) -> SelectBuilder<'a, Q, FilterEq<F, C, Field>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: FilterEq::new(self.filter, expected),
            marker: PhantomData,
        }
    }

    /// Consume the builder into a lazy, forward-only row iterator.
    #[inline]
    pub fn iter(self) -> QueryIter<'a, Q, F> {
        QueryIter {
            archetypes: self.archetypes.iter(),
            current: None,
            row: 0,
            segment_start_row: 0,
            segment_offset: 0,
            entities: [].iter(),
            filtered_mask: None,
            state: None,
            remaining_upper_bound: self.row_upper_bound,
            filter: self.filter,
        }
    }

    /// Consume the builder into an archetype-level contiguous slice iterator.
    ///
    /// Iterates over every table matching the filter predicates, yielding
    /// borrow tuples of contiguous slices `(&[EntityId], (&[A], &[B]))`.
    #[inline]
    pub fn chunks(self) -> QueryChunksIter<'a, Q, F> {
        QueryChunksIter {
            archetypes: self.archetypes.iter(),
            current: None,
            row: 0,
            filter: self.filter,
            marker: PhantomData,
        }
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> IntoIterator for SelectBuilder<'a, Q, F> {
    type Item = Q::Item;
    type IntoIter = QueryIter<'a, Q, F>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Lock-step iteration over borrowed entity and component slices.
///
/// No per-row identity lookup, downcast, allocation, clone or encoding is done.
/// Order follows current table/row order; it is not stable across structural
/// writes or snapshot restoration. Borrowing excludes writes during iteration.
pub struct QueryIter<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::Iter<'a, Archetype>,
    current: Option<&'a Archetype>,
    row: usize,
    segment_start_row: usize,
    segment_offset: usize,
    entities: std::slice::Iter<'a, EntityId>,
    filtered_mask: Option<crate::storage::archetype::RowBitMask>,
    state: Option<Q::State>,
    remaining_upper_bound: usize,
    filter: F,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> Iterator for QueryIter<'a, Q, F> {
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
            if let Some(archetype) = self.current {
                if self.row < archetype.len() {
                    if let Some((state, end)) = Q::prepare(archetype, self.row) {
                        assert!(end > self.row && end <= archetype.len(), "query segment must advance");
                        self.segment_start_row = self.row;
                        self.segment_offset = 0;
                        self.entities = archetype.entities()[self.row..end].iter();
                        self.row = end;
                        self.state = Some(state);
                        continue;
                    }
                    self.remaining_upper_bound -= archetype.len() - self.row;
                }
                self.current = None;
                self.filtered_mask = None;
            }
            let archetype = self.archetypes.next()?;
            if archetype.is_empty() {
                continue;
            }
            if self.filter.matches_archetype(archetype) {
                if F::IS_ROW_FILTERED {
                    let mask = self.filter.collect_matched_row_mask(archetype);
                    if mask.as_ref().is_some_and(|m| m.is_all_zero()) {
                        self.remaining_upper_bound -= archetype.len();
                        continue;
                    }
                    self.filtered_mask = mask;
                }
                self.current = Some(archetype);
                self.row = 0;
                continue;
            }
            self.remaining_upper_bound -= archetype.len();
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

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryIter<'a, Q, F> {}

/// Archetype-level iteration over contiguous entity and component slices.
///
/// Yields `Q::Chunk` (e.g., `(&'a [EntityId], (&'a [A], &'a [B]))`) for each matching table.
/// Slices are contiguous, equal-length, zero-copy, and suitable for SIMD or batch processing.
pub struct QueryChunksIter<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::Iter<'a, Archetype>,
    current: Option<&'a Archetype>,
    row: usize,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> Iterator for QueryChunksIter<'a, Q, F> {
    type Item = Q::Chunk;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(archetype) = self.current {
                if self.row < archetype.len() {
                    if let Some((chunk, end)) = Q::chunk(archetype, self.row) {
                        assert!(end > self.row && end <= archetype.len(), "query segment must advance");
                        if end < archetype.len() {
                            self.row = end;
                        } else {
                            self.current = None;
                        }
                        return Some(chunk);
                    }
                }
                self.current = None;
            }
            let archetype = self.archetypes.next()?;
            if archetype.is_empty() {
                continue;
            }
            if self.filter.matches_archetype(archetype) {
                if let Some((chunk, end)) = Q::chunk(archetype, 0) {
                    assert!(end > 0 && end <= archetype.len(), "query segment must advance");
                    if end < archetype.len() {
                        self.current = Some(archetype);
                        self.row = end;
                    }
                    return Some(chunk);
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // One table can now yield several physical segments.
        (0, None)
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryChunksIter<'a, Q, F> {}

/// 强类型分块查询迭代器。
///
/// 遍历包含组件 `T` 的所有 Archetype，直接借用连续切片 `(&[EntityId], &[T])`，实现完全零堆分配与零拷贝遍历。
pub struct QueryChunks<'a, T: ModelComponent + Serialize + DeserializeOwned> {
    inner: QueryChunksIter<'a, &'a T>,
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryChunks<'a, T> {
    pub(crate) fn new(archetypes: &'a [Archetype]) -> Self {
        Self {
            inner: QueryChunksIter {
                archetypes: archetypes.iter(),
                current: None,
                row: 0,
                filter: All,
                marker: PhantomData,
            },
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> Iterator for QueryChunks<'a, T> {
    type Item = (&'a [EntityId], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

