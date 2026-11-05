//! Private structural-plan storage. Inline temporary operations and shared
//! owned arrays replace per-entity maps/vectors; receipt APIs are unchanged.
use crate::storage::ErasedComponentValue;
use crate::ComponentTypeId;
use std::ops::Range;


/// Batch-owned in-place operations. Each multi-component entity owns a disjoint
/// logical range. Inline inputs share an array; already heap-owned inputs keep
/// their allocation in a batch-owned sidecar, not in every entity's plan enum.
/// Owning iterators drop unconsumed values on abandonment and unwind.
#[derive(Debug, Default)]
pub(crate) struct PlannedInPlaceValues {
    values: Vec<(ComponentTypeId, super::InPlaceOp)>,
    reserve_hint: usize,
    owned: Vec<OwnedInPlaceRange>,
    total: usize,
}

#[derive(Debug)]
struct OwnedInPlaceRange {
    start: usize,
    values: Vec<(ComponentTypeId, super::InPlaceOp)>,
    // Shared-array length at insertion. Together with start/len this gives
    // the cumulative owned length without rescanning preceding ranges.
    shared_start: usize,
}

impl PlannedInPlaceValues {
    pub(crate) fn new(reserve_hint: usize) -> Self {
        Self { values: Vec::new(), reserve_hint, owned: Vec::new(), total: 0 }
    }

    pub(crate) fn append(
        &mut self,
        ops: impl IntoIterator<Item = (ComponentTypeId, super::InPlaceOp)>,
    ) -> Range<usize> {
        let iter = ops.into_iter();
        if self.values.is_empty() {
            self.values.reserve(self.reserve_hint.max(iter.size_hint().0));
        }
        let start = self.total;
        let before = self.values.len();
        self.values.extend(iter);
        self.total += self.values.len() - before;
        start..self.total
    }

    /// Only called for an existing ordinary heap-backed input, after preflight.
    pub(crate) fn append_owned(
        &mut self,
        values: Vec<(ComponentTypeId, super::InPlaceOp)>,
    ) -> Range<usize> {
        assert!(!values.is_empty());
        // The shared-value estimate says nothing about the number of owned
        // ranges. Grow this sparse directory only as actual ranges arrive.
        let start = self.total;
        self.total += values.len();
        self.owned.push(OwnedInPlaceRange { start, values, shared_start: self.values.len() });
        start..self.total
    }

    pub(crate) fn sort_range(&mut self, range: Range<usize>) {
        // Only native/mixed input used this order in the original planner.
        // Ordinary input retains recording order (including destructor order).
        // Mixed/native ranges are appended to the shared array immediately
        // before sorting. Owned ordinary ranges retain their recording order.
        assert_eq!(range.end, self.total);
        let start = self.values.len() - range.len();
        self.values[start..].sort_unstable_by_key(|(id, _)| *id);
    }

    pub(crate) fn into_values(self) -> MixedInPlaceValuesIter {
        MixedInPlaceValuesIter {
            total: self.total, shared_total: self.values.len(), consumed_owned: 0,
            values: self.values.into_iter(),
            owned: self.owned.into_iter().peekable(),
        }
    }

    pub(crate) fn has_owned(&self) -> bool { !self.owned.is_empty() }

    /// Plans refer to whole, disjoint ranges. Directory starts are increasing;
    /// read-only preparation and undo handover may visit them in any order.
    fn locate_range(&self, range: &Range<usize>) -> (Option<usize>, Range<usize>) {
        match self.owned.binary_search_by_key(&range.start, |entry| entry.start) {
            Ok(index) => {
                assert_eq!(range.len(), self.owned[index].values.len());
                (Some(index), 0..range.len())
            }
            Err(index) => {
                let owned_prefix = index.checked_sub(1).map_or(0, |i| {
                    let entry = &self.owned[i];
                    entry.start + entry.values.len() - entry.shared_start
                });
                let start = range.start - owned_prefix;
                (None, start..start + range.len())
            }
        }
    }

    fn range_values(&self, range: &Range<usize>) -> &[(ComponentTypeId, super::InPlaceOp)] {
        if range.is_empty() { return &[]; }
        let (owned, slice) = self.locate_range(range);
        let values = owned.map_or(self.values.as_slice(), |i| self.owned[i].values.as_slice());
        &values[slice]
    }

    pub(crate) fn inspect_cids(&self, range: &Range<usize>, mut f: impl FnMut(ComponentTypeId)) {
        for (cid, _) in self.range_values(range) { f(*cid); }
    }

    pub(crate) fn for_each_range(&self, range: &Range<usize>, mut f: impl FnMut((ComponentTypeId, &super::InPlaceOp))) {
        for (cid, op) in self.range_values(range) { f((*cid, op)); }
    }

    pub(crate) fn replace_owned_value(
        &mut self,
        range: &Range<usize>,
        component: ComponentTypeId,
        real_value: ErasedComponentValue,
    ) -> bool {
        if range.is_empty() { return false; }
        let (owned, slice) = self.locate_range(range);
        let values = match owned {
            Some(i) => &mut self.owned[i].values,
            None => &mut self.values,
        };
        for (cid, op) in &mut values[slice] {
            if *cid == component {
                *op = match op {
                    super::InPlaceOp::Replace(_) => super::InPlaceOp::Replace(real_value),
                    super::InPlaceOp::Upsert(_) => super::InPlaceOp::Upsert(real_value),
                };
                return true;
            }
        }
        false
    }
}

/// Statically chosen range consumer. The shared-only kernel keeps exactly the
/// original owning iterator; it never executes segmented-owner bookkeeping.
pub(crate) trait InPlaceValueSource {
    type RangeIter<'a>: Iterator<Item = (ComponentTypeId, super::InPlaceOp)>
        where Self: 'a;
    fn from_values(values: PlannedInPlaceValues) -> Self;
    fn take_range(&mut self, range: Range<usize>) -> Self::RangeIter<'_>;

    fn for_each_range(
        &mut self,
        range: Range<usize>,
        visit: impl FnMut((ComponentTypeId, super::InPlaceOp)),
    ) {
        let mut visit = visit;
        for value in self.take_range(range) { visit(value); }
    }
}

pub(crate) struct InPlaceValuesIter {
    total: usize,
    values: std::vec::IntoIter<(ComponentTypeId, super::InPlaceOp)>,
}

impl InPlaceValueSource for InPlaceValuesIter {
    type RangeIter<'a> = std::iter::Take<&'a mut std::vec::IntoIter<(ComponentTypeId, super::InPlaceOp)>>;

    fn from_values(values: PlannedInPlaceValues) -> Self {
        assert!(!values.has_owned(), "shared consumer requires shared storage");
        Self { total: values.values.len(), values: values.values.into_iter() }
    }

    fn take_range(&mut self, range: Range<usize>) -> Self::RangeIter<'_> {
        assert_eq!(range.start, self.total - self.values.len(), "consume ranges in plan order");
        assert!(range.end <= self.total, "validated operation range");
        self.values.by_ref().take(range.len())
    }
}

pub(crate) struct MixedInPlaceValuesIter {
    total: usize,
    values: std::vec::IntoIter<(ComponentTypeId, super::InPlaceOp)>,
    shared_total: usize,
    consumed_owned: usize,
    owned: std::iter::Peekable<std::vec::IntoIter<OwnedInPlaceRange>>,
}

impl InPlaceValueSource for MixedInPlaceValuesIter {
    type RangeIter<'a> = std::iter::Chain<
        std::vec::IntoIter<(ComponentTypeId, super::InPlaceOp)>,
        std::iter::Take<&'a mut std::vec::IntoIter<(ComponentTypeId, super::InPlaceOp)>>>;

    fn from_values(values: PlannedInPlaceValues) -> Self { values.into_values() }

    fn take_range(&mut self, range: Range<usize>) -> Self::RangeIter<'_> {
        let owned = self.take_owned(&range).unwrap_or_default();
        let shared_len = if owned.is_empty() { range.len() } else { 0 };
        owned.into_iter().chain(self.values.by_ref().take(shared_len))
    }

    fn for_each_range(
        &mut self,
        range: Range<usize>,
        visit: impl FnMut((ComponentTypeId, super::InPlaceOp)),
    ) {
        // Choose once per entity, outside the component loop. Shared entities
        // in a mixed batch must not traverse an empty owned iterator per value.
        let mut owned = self.take_owned(&range).map(Vec::into_iter);
        let values = owned.as_mut().unwrap_or(&mut self.values);
        let mut visit = visit;
        for value in values.take(range.len()) { visit(value); }
    }
}

impl MixedInPlaceValuesIter {
    fn take_owned(&mut self, range: &Range<usize>)
        -> Option<Vec<(ComponentTypeId, super::InPlaceOp)>>
    {
        assert_eq!(range.start, self.shared_total - self.values.len() + self.consumed_owned,
            "consume ranges in plan order");
        assert!(range.end <= self.total, "validated operation range");
        assert!(range.start <= range.end);
        if self.owned.peek().is_some_and(|entry| entry.start == range.start) {
            let values = self.owned.next().unwrap().values;
            assert_eq!(values.len(), range.len());
            self.consumed_owned += values.len();
            Some(values)
        } else {
            assert!(self.owned.peek().is_none_or(|entry| range.end <= entry.start));
            None
        }
    }
}

#[derive(Debug)]
pub(crate) enum SmallList<T> {
    Empty,
    Single(T),
    Multiple(Vec<T>),
}

impl<T> SmallList<T> {
    pub(crate) fn new() -> Self {
        Self::Empty
    }
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Empty => &[],
            Self::Single(v) => std::slice::from_ref(v),
            Self::Multiple(v) => v,
        }
    }
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Empty => &mut [],
            Self::Single(v) => std::slice::from_mut(v),
            Self::Multiple(v) => v,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn push(&mut self, value: T) {
        match self {
            Self::Empty => *self = Self::Single(value),
            Self::Multiple(values) => values.push(value),
            Self::Single(_) => {
                let Self::Single(first) = std::mem::replace(self, Self::Empty) else {
                    unreachable!()
                };
                *self = Self::Multiple(vec![first, value]);
            }
        }
    }
}

impl SmallList<(ComponentTypeId, ErasedComponentValue)> {
    pub(crate) fn insert(&mut self, type_id: ComponentTypeId, value: ErasedComponentValue) {
        debug_assert!(!self.keys().any(|id| *id == type_id));
        self.push((type_id, value));
    }
    pub(crate) fn keys(&self) -> impl Iterator<Item = &ComponentTypeId> {
        self.as_slice().iter().map(|(id, _)| id)
    }
}

impl<T> IntoIterator for SmallList<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<std::option::IntoIter<T>, std::vec::IntoIter<T>>;
    fn into_iter(self) -> Self::IntoIter {
        let (one, rest) = match self {
            Self::Empty => (None, Vec::new()),
            Self::Single(v) => (Some(v), Vec::new()),
            Self::Multiple(v) => (None, v),
        };
        one.into_iter().chain(rest)
    }
}

impl<'a, T> IntoIterator for &'a SmallList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

#[derive(Debug)]
pub(crate) struct PlannedPatchValues {
    values: Vec<(ComponentTypeId, Option<ErasedComponentValue>)>,
    removed: Vec<ComponentTypeId>,
    pool: Vec<crate::storage::batch::PoolEntry>,
    recycled_plans: Vec<super::PlannedOp>,
    reserve_hint: usize,
    has_drop_values: bool,
}

impl Drop for PlannedPatchValues {
    fn drop(&mut self) {
        if !self.has_drop_values {
            unsafe {
                self.values.set_len(0);
                self.pool.set_len(0);
            }
        }
    }
}

impl PlannedPatchValues {
    pub(crate) fn new(reserve_hint: usize) -> Self {
        Self {
            values: Vec::new(),
            removed: Vec::new(),
            pool: Vec::new(),
            recycled_plans: Vec::new(),
            reserve_hint,
            has_drop_values: false,
        }
    }

    #[inline]
    pub(crate) fn set_pool(
        &mut self,
        pool: Vec<crate::storage::batch::PoolEntry>,
        pool_has_drop: bool,
    ) {
        self.pool = pool;
        self.has_drop_values |= pool_has_drop;
    }

    #[inline]
    pub(crate) fn pool_slice_mut(
        &mut self,
        start: usize,
        len: usize,
    ) -> &mut [crate::storage::batch::PoolEntry] {
        &mut self.pool[start..start + len]
    }

    #[inline]
    pub(crate) fn recycle_plans(&mut self, plans: Vec<super::PlannedOp>) {
        self.recycled_plans = plans;
    }

    /// Input is sorted by type id by the planner. Ranges never overlap.
    pub(crate) fn append_values(
        &mut self,
        values: impl IntoIterator<Item = (ComponentTypeId, ErasedComponentValue)>,
    ) -> Range<usize> {
        let iter = values.into_iter();
        let lower = iter.size_hint().0;
        if self.values.is_empty() && lower != 0 {
            self.values.reserve(self.reserve_hint.max(lower));
        }
        let start = self.values.len();
        let mut has_drop = self.has_drop_values;
        self.values.extend(iter.map(|(id, value)| {
            if value.needs_drop() {
                has_drop = true;
            }
            (id, Some(value))
        }));
        self.has_drop_values = has_drop;
        let range = start..self.values.len();
        debug_assert!(
            self.values[range.clone()]
                .windows(2)
                .all(|w| w[0].0 < w[1].0)
        );
        range
    }

    pub(crate) fn append_removed(&mut self, types: SmallList<ComponentTypeId>) -> Range<usize> {
        if self.removed.is_empty() && types.len() != 0 {
            self.removed.reserve(self.reserve_hint.max(types.len()));
        }
        let start = self.removed.len();
        self.removed.extend(types);
        start..self.removed.len()
    }

    #[inline]
    pub(crate) fn take_slice_mut(
        &mut self,
        range: &Range<usize>,
    ) -> &mut [(ComponentTypeId, Option<ErasedComponentValue>)] {
        &mut self.values[range.clone()]
    }

    #[inline]
    pub(crate) fn take(
        &mut self,
        range: &Range<usize>,
        type_id: ComponentTypeId,
    ) -> Option<ErasedComponentValue> {
        let values = &mut self.values[range.clone()];
        let index = values.binary_search_by_key(&type_id, |(id, _)| *id).ok()?;
        values[index].1.take()
    }

    #[inline]
    pub(crate) fn replace_value(
        &mut self,
        range: &Range<usize>,
        type_id: ComponentTypeId,
        real_value: ErasedComponentValue,
    ) -> bool {
        if range.end <= self.values.len() {
            let values = &mut self.values[range.clone()];
            if let Ok(index) = values.binary_search_by_key(&type_id, |(id, _)| *id) {
                if real_value.needs_drop() {
                    self.has_drop_values = true;
                }
                values[index].1 = Some(real_value);
                return true;
            }
        }
        false
    }

    #[inline]
    pub(crate) fn is_placeholder(&self, range: &Range<usize>, type_id: ComponentTypeId) -> bool {
        if range.end <= self.values.len() {
            let values = &self.values[range.clone()];
            if let Ok(index) = values.binary_search_by_key(&type_id, |(id, _)| *id) {
                return values[index].1.as_ref().is_some_and(|v| v.is_placeholder());
            }
        }
        false
    }

    pub(crate) fn removed(&self, range: Range<usize>) -> &[ComponentTypeId] {
        &self.removed[range]
    }

    pub(crate) fn value_cids(&self, range: Range<usize>) -> impl Iterator<Item = ComponentTypeId> + '_ {
        self.values[range].iter().map(|(id, _)| *id)
    }

    pub(crate) fn inspect_cids(&self, range: &Range<usize>, mut f: impl FnMut(ComponentTypeId)) {
        if range.end <= self.values.len() {
            for (cid, _) in &self.values[range.clone()] {
                f(*cid);
            }
        }
    }

    pub(crate) fn drain_range(
        &mut self,
        range: Range<usize>,
    ) -> Vec<(ComponentTypeId, ErasedComponentValue)> {
        let mut out = Vec::with_capacity(range.len());
        for item in &mut self.values[range] {
            if let Some(val) = item.1.take() {
                out.push((item.0, val));
            }
        }
        out
    }

    pub(crate) fn removed_range(&self, range: Range<usize>) -> &[ComponentTypeId] {
        &self.removed[range]
    }
}

#[cfg(test)]
#[path = "planned_patch_values_tests.rs"]
mod tests;
