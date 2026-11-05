use super::*;
use crate::storage::core::store::LayoutOptimization;

#[test]
fn test_store_crud_layout_optimization_and_reorder() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::FastWrite);
    assert_eq!(store.layout_optimization(), LayoutOptimization::FastWrite);
    assert_eq!(store.optimize_layout(), 0);

    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);
    let e3 = EntityId::from_u128(3);
    store.insert((e1, (10u64, "a".to_string()))).unwrap();
    store.insert((e2, (20u64, "b".to_string()))).unwrap();
    store.insert((e3, (30u64, "c".to_string()))).unwrap();

    let moved = store.reorder_entities(&[e3, e1, e2]).unwrap();
    assert!(moved > 0);
    assert_eq!(store.get::<u64>(e1), Some(&10));
    assert_eq!(store.get::<u64>(e2), Some(&20));
    assert_eq!(store.get::<u64>(e3), Some(&30));
}

#[test]
fn test_store_singleton_and_partition_merge() {
    let mut store = Store::new();
    assert!(!store.has_singleton::<u64>());
    assert_eq!(store.set_singleton(42u64).unwrap(), None);
    assert!(store.has_singleton::<u64>());
    assert_eq!(store.singleton::<u64>(), Some(&42));
    *store.singleton_mut::<u64>().unwrap() = 99;
    assert_eq!(store.remove_singleton::<u64>().unwrap(), Some(99));
    assert!(!store.has_singleton::<u64>());

    let e1 = EntityId::from_u128(10);
    store.insert((e1, (100u64,))).unwrap();

    let mut forked = store.fork_empty();
    let e2 = EntityId::from_u128(20);
    forked.insert((e2, (200u64,))).unwrap();

    assert!(store.can_merge_partition(&forked));
    store.merge_partition(forked).unwrap();
    assert_eq!(store.get::<u64>(e1), Some(&100));
    assert_eq!(store.get::<u64>(e2), Some(&200));
}
