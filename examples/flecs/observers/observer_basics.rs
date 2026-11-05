//! Port of `examples/flecs/observers/observer_basics.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

fn set_position_observed(store: &mut Store, entity: EntityId, pos: Position) {
    let was_present = store.has::<Position>(entity);
    store.set(entity, pos).unwrap();
    let name = store.name_of(entity);
    if !was_present {
        elog!(" - OnAdd: Position: {}", name);
    }
    if let Some(current) = store.get::<Position>(entity) {
        elog!(" - OnSet: Position: {}: with {:?}", name, current);
    }
}

fn remove_position_observed(store: &mut Store, entity: EntityId) {
    if let Ok(Some(removed)) = store.remove::<Position>(entity) {
        let name = store.name_of(entity);
        elog!(" - OnRemove: Position: {}: with {:?}", name, removed);
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity, set Position (emits OnAdd and OnSet)
    let entity = store.entity_named("e1");
    set_position_observed(&mut store, entity, Position { x: 10.0, y: 20.0 });

    // Remove Position (emits OnRemove)
    remove_position_observed(&mut store, entity);

    // Remove Position again (no event emitted because component is no longer present)
    remove_position_observed(&mut store, entity);
}

#[test]
fn test_observer_basics() {
    main();
    assert_eq!(
        take_log(),
        vec![
            " - OnAdd: Position: e1",
            " - OnSet: Position: e1: with Position { x: 10.0, y: 20.0 }",
            " - OnRemove: Position: e1: with Position { x: 10.0, y: 20.0 }",
        ]
    );
}
