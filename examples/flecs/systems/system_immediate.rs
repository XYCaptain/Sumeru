//! Port of `examples/flecs/systems/system_immediate.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Waiter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Plate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AssignedPlate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AssignedWaiter;

fn assign_plate(store: &mut Store, waiter: EntityId, plate: EntityId) {
    store.set_pair(waiter, Plate, plate).unwrap();
    store.set(waiter, AssignedPlate).unwrap();
}

fn assign_waiter(store: &mut Store, plate: EntityId, waiter: EntityId) {
    store.set_pair(plate, Waiter, waiter).unwrap();
    store.set(plate, AssignedWaiter).unwrap();
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let waiter_1 = store.entity_named("waiter_1");
    store.set(waiter_1, Waiter).unwrap();
    let waiter_2 = store.entity_named("waiter_2");
    store.set(waiter_2, Waiter).unwrap();
    let waiter_3 = store.entity_named("waiter_3");
    store.set(waiter_3, Waiter).unwrap();

    let plate_1 = store.entity_named("plate_1");
    store.set(plate_1, Plate).unwrap();
    let plate_2 = store.entity_named("plate_2");
    store.set(plate_2, Plate).unwrap();
    let plate_3 = store.entity_named("plate_3");
    store.set(plate_3, Plate).unwrap();

    assign_plate(&mut store, waiter_1, plate_2);
    assign_waiter(&mut store, plate_2, waiter_1);

    // Collect unassigned plates (`With<Plate>`, `Without<AssignedWaiter>`)
    let unassigned_plates: Vec<EntityId> = store
        .query::<&Plate>()
        .without::<AssignedWaiter>()
        .iter()
        .map(|(e, _)| e)
        .collect();

    // Immediate system: assigns plates directly to available waiters so the same waiter
    // is never picked twice, while deferring mutations on the iterated plate entity.
    let mut deferred_plate_assignments = Vec::new();
    for plate in unassigned_plates {
        let available_waiter = store
            .query::<&Waiter>()
            .without::<AssignedPlate>()
            .iter()
            .map(|(w, _)| w)
            .next();

        if let Some(waiter) = available_waiter {
            // Immediate mutation on `waiter` (analogous to `defer_suspend` / `defer_resume`)
            assign_plate(&mut store, waiter, plate);

            // Queue deferred mutation on `plate`
            deferred_plate_assignments.push((plate, waiter));

            elog!(
                "Assigned {} to {}!",
                store.name_of(waiter),
                store.name_of(plate)
            );
        }
    }
    let mut cmd = store.commands();
    for (plate, waiter) in deferred_plate_assignments {
        cmd.entity(plate)
            .set(Pair::new(Waiter, waiter))
            .set(AssignedWaiter);
    }
    cmd.apply_discard(&mut store).unwrap();
}

#[test]
fn test_system_immediate() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Assigned waiter_3 to plate_1!",
            "Assigned waiter_2 to plate_3!",
        ]
    );
}
