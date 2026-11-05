use super::*;

#[test]
fn in_place_ranges_preserve_recording_order_and_consume_once() {
    let mut plan = PlannedInPlaceValues::new(4);
    let mut ids = [ComponentTypeId::of::<u64>(), ComponentTypeId::of::<String>()];
    ids.sort_unstable();
    let make = |id, value: &str| (id, super::super::InPlaceOp::Replace(
        ErasedComponentValue::with_component_type_id(id, value.to_owned()),
    ));
    let first = plan.append([make(ids[1], "first"), make(ids[0], "second")]);
    let next = plan.append([make(ids[0], "third")]);
    assert_eq!(first, 0..2);
    assert_eq!(next, 2..3);
    let mut plan = plan.into_values();
    let values: Vec<_> = plan.take_range(first).collect();
    assert_eq!(values[0].0, ids[1]);
    assert_eq!(values[1].0, ids[0]);
    let third = plan.take_range(next).next().unwrap().1.into_value();
    assert_eq!(third.downcast_ref::<String>().unwrap(), "third");
    assert_eq!(plan.values.len(), 0);
}

#[test]
fn temporary_single_is_inline_and_multiple_retains_order() {
    let mut values = SmallList::new();
    assert!(values.is_empty());
    values.push(3);
    assert!(matches!(values, SmallList::Single(3)));
    values.push(1);
    values.push(2);
    values.as_mut_slice().sort_unstable();
    assert_eq!(values.into_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(
        SmallList::Single(4).into_iter().collect::<Vec<_>>(),
        vec![4]
    );
}

#[test]
fn owned_ranges_interleave_with_shared_ranges_without_copying() {
    let ty = ComponentTypeId::of::<String>();
    let make = |s: &str| (ty, super::super::InPlaceOp::Replace(
        ErasedComponentValue::new(s.to_owned()),
    ));
    let mut plan = PlannedInPlaceValues::new(8);
    let a = plan.append([make("a")]);
    let owned = vec![make("b"), make("c"), make("d")];
    let address = owned.as_ptr();
    let b = plan.append_owned(owned);
    assert_eq!(plan.owned[0].values.as_ptr(), address);
    let c = plan.append([make("e"), make("f")]);
    plan.sort_range(c.clone());
    let d = plan.append_owned(vec![make("g")]);
    assert_eq!([a.clone(), b.clone(), c.clone(), d.clone()], [0..1, 1..4, 4..6, 6..7]);
    let mut values = plan.into_values();
    let mut output = Vec::new();
    for range in [a,b,c,d] {
        values.for_each_range(range, |(_, op)| {
            output.push(op.into_value().into_typed::<String>().unwrap());
        });
    }
    assert_eq!(output, ["a","b","c","d","e","f","g"]);
    assert_eq!(values.consumed_owned + values.shared_total - values.values.len(), 7);
    assert!(values.owned.next().is_none());
    assert_eq!(values.values.len(), 0);
}

#[test]
fn indexed_ranges_support_reverse_inspection_and_handover() {
    let ty = ComponentTypeId::of::<u64>();
    let make = |n| (ty, super::super::InPlaceOp::Upsert(ErasedComponentValue::new(n)));
    let mut plan = PlannedInPlaceValues::new(128);
    let mut ranges = Vec::new();
    for i in 0..129u64 {
        ranges.push(if i % 3 == 0 {
            plan.append_owned(vec![make(i)])
        } else { plan.append([make(i)]) });
    }
    for (i, range) in ranges.iter().enumerate().rev() {
        let mut ids = Vec::new();
        plan.inspect_cids(range, |cid| ids.push(cid));
        assert_eq!(ids, [ty]);
        plan.for_each_range(range, |(cid, op)| {
            assert_eq!(cid, ty);
            let super::super::InPlaceOp::Upsert(value) = op else { panic!("keep operation kind") };
            assert_eq!(value.downcast_ref::<u64>(), Some(&(i as u64)));
        });
        assert!(!plan.replace_owned_value(range, ComponentTypeId::of::<u32>(), ErasedComponentValue::new(0u32)));
        assert!(plan.replace_owned_value(range, ty, ErasedComponentValue::new(i as u64 + 1000)));
    }
    plan.inspect_cids(&(0..0), |_| panic!("empty range"));
    let mut consumer = plan.into_values();
    for (i, range) in ranges.into_iter().enumerate() {
        consumer.for_each_range(range, |(_, op)| {
            assert!(matches!(&op, super::super::InPlaceOp::Upsert(_)));
            assert_eq!(op.into_value().into_typed::<u64>().unwrap(), i as u64 + 1000);
        });
    }
}

#[test]
fn sparse_owned_directory_does_not_use_shared_capacity_hint() {
    let mut plan = PlannedInPlaceValues::new(2_000_000);
    let ty = ComponentTypeId::of::<String>();
    let values = vec![(ty, super::super::InPlaceOp::Replace(
        ErasedComponentValue::new(String::from("owned")),
    ))];
    let address = values.as_ptr();
    assert_eq!(plan.append_owned(values), 0..1);
    assert_eq!(plan.owned[0].values.as_ptr(), address);
    assert!(plan.owned.capacity() <= 4);
    assert_eq!(plan.values.capacity(), 0);
}

#[test]
fn range_visitor_unwind_drops_shared_and_owned_values_once() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct Probe {
        #[serde(skip)]
        drops: Option<Arc<AtomicUsize>>,
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            if let Some(drops) = &self.drops { drops.fetch_add(1, Ordering::SeqCst); }
        }
    }
    // Panic in each of the shared/owned/shared ranges, including while
    // owned values and subsequent shared values remain unconsumed.
    for panic_range in 0..3 {
        let drops = Arc::new(AtomicUsize::new(0));
        let make = || (ComponentTypeId::of::<Probe>(), super::super::InPlaceOp::Replace(
            ErasedComponentValue::new(Probe { drops: Some(drops.clone()) }),
        ));
        let mut plan = PlannedInPlaceValues::new(4);
        let ranges = [plan.append([make(), make()]),
            plan.append_owned(vec![make(), make(), make()]),
            plan.append([make(), make()])];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut values = plan.into_values();
            for (index, range) in ranges.into_iter().enumerate() {
                values.for_each_range(range, |value| {
                    drop(value);
                    assert_ne!(index, panic_range, "injected visitor panic");
                });
            }
        }));
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 7);
    }
}

#[test]
fn shared_ranges_are_disjoint_and_values_are_taken_once() {
    let mut plan = PlannedPatchValues::new(8);
    let type_id = ComponentTypeId::of::<String>();
    let range = plan.append_values(SmallList::Single((
        type_id,
        ErasedComponentValue::with_component_type_id(type_id, String::from("first")),
    )));
    let next = plan.append_values(SmallList::Single((
        type_id,
        ErasedComponentValue::with_component_type_id(type_id, String::from("second")),
    )));
    assert_eq!(range, 0..1);
    assert_eq!(next, 1..2);
    assert!(plan.take(&range, ComponentTypeId::of::<u64>()).is_none());
    assert_eq!(
        plan.take(&range, type_id)
            .unwrap()
            .downcast_ref::<String>()
            .unwrap(),
        "first"
    );
    assert!(plan.take(&range, type_id).is_none());
    assert_eq!(
        plan.take(&next, type_id)
            .unwrap()
            .downcast_ref::<String>()
            .unwrap(),
        "second"
    );
    let removed = plan.append_removed(SmallList::Single(type_id));
    assert_eq!(plan.removed(removed), &[type_id]);
}
