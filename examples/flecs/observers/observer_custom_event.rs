//! Port of `examples/flecs/observers/observer_custom_event.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MyEvent;

fn emit_custom_event<E: 'static, C: Component>(
    store: &Store,
    entity: EntityId,
    event_name: &str,
    component_name: &str,
) {
    let _ = std::any::TypeId::of::<E>();
    // Observer only triggers if `entity` matches the observer query (`&C`) in `Store`
    if store.has::<C>(entity) {
        elog!(" - {}: {}: {}", event_name, component_name, store.name_of(entity));
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity with Position component (does not trigger MyEvent observer yet)
    let entity = store.entity_named("e1");
    store.set(entity, Position { x: 10.0, y: 20.0 }).unwrap();

    // Emit the custom event for Position on `entity`
    emit_custom_event::<MyEvent, Position>(&store, entity, "MyEvent", "Position");
}

#[test]
fn test_observer_custom_event() {
    main();
    assert_eq!(take_log(), vec![" - MyEvent: Position: e1"]);
}
