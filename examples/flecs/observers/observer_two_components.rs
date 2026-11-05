//! Port of `examples/flecs/observers/observer_two_components.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

fn notify_two_components(store: &Store, entity: EntityId, set_component: &str) {
    if let (Some(pos), Some(vel)) = (
        store.get::<Position>(entity),
        store.get::<Velocity>(entity),
    ) {
        elog!(
            " - OnSet: {}: {}: p: {{ {}, {} }}, v: {{ {}, {} }}",
            set_component,
            store.name_of(entity),
            pos.x,
            pos.y,
            vel.x,
            vel.y
        );
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity, set Position (does not yet match (&Position, &Velocity) observer)
    let entity = store.entity_named("e");
    store.set(entity, Position { x: 10.0, y: 20.0 }).unwrap();
    notify_two_components(&store, entity, "Position");

    // Set Velocity (now matches (&Position, &Velocity) observer)
    store.set(entity, Velocity { x: 1.0, y: 2.0 }).unwrap();
    notify_two_components(&store, entity, "Velocity");
}

#[test]
fn test_observer_two_components() {
    main();
    assert_eq!(
        take_log(),
        vec![" - OnSet: Velocity: e: p: { 10, 20 }, v: { 1, 2 }"]
    );
}
