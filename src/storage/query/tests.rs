use super::*;
use crate::{EntityBundle, Store, StoreBatch};
use std::cell::Cell;

thread_local! {
    static PREPARES: Cell<usize> = const { Cell::new(0) };
    static FETCHES: Cell<usize> = const { Cell::new(0) };
}

// Private instrumentation proves pruning happens before slice extraction,
// not merely after visiting rows. External callers cannot implement Fetch.
struct Counted;
impl<'a> QueryData<'a> for Counted {
    type Item = (EntityId, &'a u64);
    type Chunk = (&'a [EntityId], &'a [u64]);
}
impl<'a> sealed::Fetch<'a> for Counted {
    type State = ColumnIter<'a, u64>;
    fn prepare(archetype: &'a Archetype, start: usize) -> Option<(Self::State, usize)> {
        PREPARES.with(|count| count.set(count.get() + 1));
        <&u64 as sealed::Fetch>::prepare(archetype, start)
    }
    fn next(state: &mut Self::State, entity: EntityId) -> <Self as QueryData<'a>>::Item {
        FETCHES.with(|count| count.set(count.get() + 1));
        <&u64 as sealed::Fetch>::next(state, entity)
    }
    fn chunk(
        archetype: &'a Archetype,
        start: usize,
    ) -> Option<(<Self as QueryData<'a>>::Chunk, usize)> {
        <&u64 as sealed::Fetch>::chunk(archetype, start)
    }
}

#[test]
fn pruning_skips_slice_preparation_and_every_row_of_excluded_tables() {
    let mut store = Store::new();
    let mut batch = StoreBatch::new();
    for n in 1..=10_000u64 {
        let mut bundle = EntityBundle::new().with(n).unwrap();
        if n % 2 == 0 {
            bundle.insert(true).unwrap();
        }
        if n % 5 == 0 {
            bundle.insert(String::new()).unwrap();
        }
        batch
            .create_entity(EntityId::from_u128(n as u128), bundle)
            .unwrap();
    }
    store.apply_batch(batch).unwrap();
    PREPARES.with(|n| n.set(0));
    FETCHES.with(|n| n.set(0));
    let rows = store
        .query::<Counted>()
        .without::<bool>()
        .without::<String>()
        .with::<u64>()
        .iter();
    assert_eq!(PREPARES.with(Cell::get), 0, "building an iterator is lazy");
    assert_eq!(rows.count(), 4000);
    assert_eq!(
        PREPARES.with(Cell::get),
        1,
        "extract columns once for the only matching table"
    );
    assert_eq!(
        FETCHES.with(Cell::get),
        4000,
        "no fetches from the excluded 6000 rows"
    );
    PREPARES.with(|n| n.set(0));
    FETCHES.with(|n| n.set(0));
    assert_eq!(store.query::<Counted>().with::<u128>().iter().count(), 0);
    assert_eq!(PREPARES.with(Cell::get), 0);
    assert_eq!(FETCHES.with(Cell::get), 0);
}

#[test]
fn unequal_columns_cannot_silently_zip_truncate_a_table() {
    let mut store = Store::new();
    for n in 1..=2 {
        store
            .create_entity(
                EntityId::from_u128(n),
                EntityBundle::new()
                    .with(n as u64)
                    .unwrap()
                    .with(true)
                    .unwrap(),
            )
            .unwrap();
    }
    let archetype = store.archetypes.iter_mut().find(|a| !a.is_empty()).unwrap();
    archetype
        .columns_mut()
        .get_mut(&ComponentTypeId::of::<u64>())
        .unwrap()
        .as_any_mut()
        .downcast_mut::<TypedColumn<u64>>()
        .unwrap()
        .swap_remove(0);
    assert!(store.validate_invariants().is_err());
    let mut rows = store.query::<(&u64, &bool)>().iter();
    assert!(rows.next().is_none());
    assert_eq!(rows.size_hint(), (0, Some(0)));
}

#[test]
fn test_row_level_filter_read_and_write_exclusion_guarantee() {
    let mut store = Store::new();
    let e0 = EntityId::from_u128(100);
    let e1 = EntityId::from_u128(101);

    // 在同一原型中创建两个实体，u64 分别为 0 和 1
    store
        .create_entity(e0, EntityBundle::new().with(0u64).unwrap())
        .unwrap();
    store
        .create_entity(e1, EntityBundle::new().with(1u64).unwrap())
        .unwrap();

    // 1. 只读查询：确保只遍历满足 *v > 0 的实体 [1]，排除 [0]
    let items: Vec<u64> = store
        .query::<(&u64,)>()
        .filter::<u64>(|v| *v > 0)
        .iter()
        .map(|(_, (v,))| *v)
        .collect();
    assert_eq!(items, vec![1]);

    // 2. 可写查询：仅满足条件的实体获得可变引用并修改，排除实体绝对不被可变借出
    for (_, (v,)) in store
        .query_mut::<(&mut u64,)>()
        .unwrap()
        .filter::<u64>(|v| *v > 0)
        .iter_mut()
    {
        *v += 110;
    }

    // 3. 验证底层数据物理状态：e0 绝对未被修改（仍为 0），e1 顺利累加（1 + 110 = 111）
    let v0: &u64 = store.get(e0).unwrap();
    assert_eq!(*v0, 0, "排除实体绝对不可被写入修改");

    let v1: &u64 = store.get(e1).unwrap();
    assert_eq!(*v1, 111, "匹配实体完成修改");
}

/// Directly test the row-selection kernel.
/// The capacity bound applies only to matched-row metadata.
#[test]
fn lazy_index_allocation_zero_match_no_heap_sparse_match_bounded_capacity() {
    let mut store = Store::new();
    let mut batch = StoreBatch::new();
    for n in 1u64..=1000 {
        batch
            .create_entity(
                EntityId::from_u128(n as u128),
                EntityBundle::new().with(n).unwrap(),
            )
            .unwrap();
    }
    store.apply_batch(batch).unwrap();
    let arch = store
        .archetypes
        .iter()
        .find(|arch| !arch.is_empty())
        .unwrap();
    let none = BusinessFilter::<All, u64>::new(All, |v| *v > 9999);
    assert_eq!(none.collect_matched_rows(arch).unwrap().capacity(), 0);
    let sparse = BusinessFilter::<All, u64>::new(All, |v| *v <= 10);
    let rows = sparse.collect_matched_rows(arch).unwrap();
    assert_eq!(rows, (0..10).collect::<Vec<_>>());
    assert!(
        rows.capacity() <= 256,
        "sparse metadata must not reserve the table size"
    );
    let dense = BusinessFilter::<All, u64>::new(All, |v| *v >= 1);
    assert_eq!(
        dense.collect_matched_rows(arch).unwrap(),
        (0..1000).collect::<Vec<_>>()
    );

    let mut values: Vec<_> = store
        .query::<&u64>()
        .filter::<u64>(|v| *v >= 1)
        .iter()
        .map(|(_, v)| *v)
        .collect();
    values.sort();
    assert_eq!(values, (1..=1000).collect::<Vec<_>>());
}

#[test]
fn optional_query_and_query_mut_support_mixed_archetypes_in_iter_and_chunks() {
    let mut store = Store::new();
    for i in 0..20u64 {
        let id = EntityId::from_u128((i + 1) as u128);
        if i % 2 == 0 {
            store.insert((id, (i, (i * 10) as u32))).unwrap();
        } else {
            store.insert((id, (i,))).unwrap();
        }
    }

    // 1. Read-only query iter with Option<&u32>
    let mut with_u32 = 0usize;
    let mut without_u32 = 0usize;
    for (_id, (&u, opt_v)) in store.query::<(&u64, Option<&u32>)>().iter() {
        if let Some(&v) = opt_v {
            assert_eq!(v as u64, u * 10);
            with_u32 += 1;
        } else {
            assert_eq!(u % 2, 1);
            without_u32 += 1;
        }
    }
    assert_eq!(with_u32, 10);
    assert_eq!(without_u32, 10);

    // 2. Read-only query chunks with Option<&u32>
    let mut chunk_with = 0usize;
    let mut chunk_without = 0usize;
    for (ids, (u_slice, opt_v_slice)) in store.query::<(&u64, Option<&u32>)>().chunks() {
        assert_eq!(ids.len(), u_slice.len());
        if let Some(v_slice) = opt_v_slice {
            assert_eq!(v_slice.len(), u_slice.len());
            chunk_with += ids.len();
        } else {
            chunk_without += ids.len();
        }
    }
    assert_eq!(chunk_with, 10);
    assert_eq!(chunk_without, 10);

    // 3. Mutable query_mut chunks with (&mut u64, Option<&u32>)
    for (ids, (u_slice, opt_v_slice)) in store
        .query_mut::<(&mut u64, Option<&u32>)>()
        .unwrap()
        .chunks()
    {
        assert_eq!(ids.len(), u_slice.len());
        if let Some(v_slice) = opt_v_slice {
            for (u, &v) in u_slice.iter_mut().zip(v_slice.iter()) {
                *u += v as u64;
            }
        } else {
            for u in u_slice.iter_mut() {
                *u += 1;
            }
        }
    }

    // 4. Mutable query_mut iter with (&u64, Option<&mut u32>)
    for (_id, (&u, opt_v)) in store
        .query_mut::<(&u64, Option<&mut u32>)>()
        .unwrap()
        .iter()
    {
        if let Some(v) = opt_v {
            *v += u as u32;
        }
    }

    for i in 0..20u64 {
        let id = EntityId::from_u128((i + 1) as u128);
        let u = *store.get::<u64>(id).unwrap();
        if i % 2 == 0 {
            assert_eq!(u, i + i * 10);
            let v = *store.get::<u32>(id).unwrap();
            assert_eq!(v as u64, i * 10 + u);
        } else {
            assert_eq!(u, i + 1);
            assert!(store.get::<u32>(id).is_none());
        }
    }
}
