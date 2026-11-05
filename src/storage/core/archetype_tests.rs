use super::*;
use crate::storage::column::ColumnBlock;
use crate::storage::core::column::backing::{ColumnBacking, DirectBlock};
use crate::{ArchetypeId, ComponentTypeId, EntityId};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn clone_failure_does_not_publish_an_earlier_column() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static PANIC_AT: AtomicUsize = AtomicUsize::new(3);
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Value<const N: u8>(i64);
    impl<const N: u8> Clone for Value<N> {
        fn clone(&self) -> Self {
            let count = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            assert_ne!(
                count,
                PANIC_AT.load(Ordering::Relaxed),
                "injected second-column Clone failure"
            );
            Self(self.0)
        }
    }
    let mut arch = Archetype::new(
        ArchetypeId(0),
        BTreeSet::from([
            ComponentTypeId::of::<Value<0>>(),
            ComponentTypeId::of::<Value<1>>(),
        ]),
    );
    let mut a = TypedColumn::<Value<0>>::new(std::any::type_name::<Value<0>>());
    let mut b = TypedColumn::<Value<1>>::new(std::any::type_name::<Value<1>>());
    for i in 0..2 {
        a.push(Value(i));
        b.push(Value(i));
        arch.push_entity(EntityId::from_u128(i as u128 + 1));
    }
    let held_a = a.freeze_column().unwrap();
    let held_b = b.freeze_column().unwrap();
    arch.columns_mut()
        .insert(ComponentTypeId::of::<Value<0>>(), Box::new(a));
    arch.columns_mut()
        .insert(ComponentTypeId::of::<Value<1>>(), Box::new(b));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arch.co_materialize_segmented_columns()
    }));
    assert!(failed.is_err());
    assert_eq!(
        CALLS.load(Ordering::Relaxed),
        3,
        "first column was prepared before the failure"
    );
    assert_eq!(arch.entities().len(), 2);
    PANIC_AT.store(usize::MAX, Ordering::Relaxed);
    arch.co_materialize_segmented_columns().unwrap();
    assert!(
        arch.get_column::<Value<0>>().unwrap().backing().is_owned(),
        "prepared column is now owned"
    );
    assert!(
        arch.get_column::<Value<1>>().unwrap().backing().is_owned(),
        "prepared column is now owned"
    );
    assert_eq!(
        held_a
            .as_any()
            .downcast_ref::<ColumnBlock<Value<0>>>()
            .unwrap()
            .as_slice()[1]
            .0,
        1
    );
    assert_eq!(
        held_b
            .as_any()
            .downcast_ref::<ColumnBlock<Value<1>>>()
            .unwrap()
            .as_slice()[1]
            .0,
        1
    );
}

#[test]
fn unique_frozen_column_materialization_moves_without_clone() {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct NoClone;
    impl Clone for NoClone {
        fn clone(&self) -> Self {
            panic!("unique data must move");
        }
    }
    let mut col = TypedColumn::<NoClone>::new(std::any::type_name::<NoClone>());
    col.push(NoClone);
    drop(col.freeze_column().unwrap());
    assert!(col.prepare_contiguous().unwrap().is_none());
    let mut arch = Archetype::new(
        ArchetypeId(0),
        BTreeSet::from([ComponentTypeId::of::<NoClone>()]),
    );
    arch.push_entity(EntityId::from_u128(1));
    arch.columns_mut()
        .insert(ComponentTypeId::of::<NoClone>(), Box::new(col));
    arch.co_materialize_segmented_columns().unwrap();
    let restored = arch.get_column::<NoClone>().unwrap();
    assert_eq!(restored.len(), 1);
    assert!(matches!(
        restored.backing(),
        ColumnBacking::Direct(DirectBlock::Owned(_))
    ));
}
