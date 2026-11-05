//! Port of `examples/flecs/observers/observer_propagate.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct ChildOf;

fn notify_onset_position(store: &Store, target: EntityId) {
    // Check if `target` itself has a parent with Position
    if let (Some(pos_self), Some(parent)) = (
        store.get::<Position>(target),
        store.pair_target::<ChildOf>(target),
    ) {
        if let Some(pos_parent) = store.get::<Position>(parent) {
            elog!(
                " - OnSet: Position: {}: self: {{ {}, {} }}, parent: {{ {}, {} }}",
                store.name_of(target),
                pos_self.x,
                pos_self.y,
                pos_parent.x,
                pos_parent.y
            );
        }
    }

    // Propagate OnSet(Position) downwards to children of `target`
    for child in store.sources_with_pair::<ChildOf>(target) {
        if let (Some(pos_self), Some(pos_parent)) =
            (store.get::<Position>(child), store.get::<Position>(target))
        {
            elog!(
                " - OnSet: Position: {}: self: {{ {}, {} }}, parent: {{ {}, {} }}",
                store.name_of(child),
                pos_self.x,
                pos_self.y,
                pos_parent.x,
                pos_parent.y
            );
        }
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity and parent
    let parent = store.entity_named("p");
    let entity = store.entity_named("e");
    store.set_pair(entity, ChildOf, parent).unwrap();

    // Set Position on entity. Does not trigger observer yet since parent has no Position.
    store.set(entity, Position { x: 10.0, y: 20.0 }).unwrap();
    notify_onset_position(&store, entity);

    // Set Position on parent. Propagates along ChildOf edge and triggers observer for `e`.
    store.set(parent, Position { x: 1.0, y: 2.0 }).unwrap();
    notify_onset_position(&store, parent);
}

#[test]
fn test_observer_propagate() {
    main();
    assert_eq!(
        take_log(),
        vec![" - OnSet: Position: e: self: { 10, 20 }, parent: { 1, 2 }"]
    );
}
