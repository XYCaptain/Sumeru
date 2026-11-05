//! Port of `examples/flecs/observers/observer_yield_existing.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create existing entities with Position component
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 20.0, y: 30.0 }).unwrap();

    // Register observer with `yield_existing`: immediately yields OnSet events for all
    // existing entities in `Store` that match `&Position`.
    for (entity, pos) in store.query::<&Position>().iter() {
        elog!(
            " - OnSet: Position: {}: {{ {}, {} }}",
            store.name_of(entity),
            pos.x,
            pos.y
        );
    }
}

#[test]
fn test_observer_yield_existing() {
    main();
    assert_eq!(
        take_log(),
        vec![
            " - OnSet: Position: e1: { 10, 20 }",
            " - OnSet: Position: e2: { 20, 30 }",
        ]
    );
}
