//! Port of `examples/flecs/entities/entity_hooks.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EntityTag;

fn set_position_with_hooks(store: &mut Store, entity: EntityId, pos: Position) {
    let is_new = !store.has::<Position>(entity);
    let name = store.name_of(entity);
    if is_new {
        elog!("added Position to {name:?}");
    }
    store.set(entity, pos).unwrap();
    if let Some(current) = store.get::<Position>(entity) {
        elog!("set {current:?} for {name:?}");
    }
}

fn destruct_with_hooks(store: &mut Store, entity: EntityId) {
    let name = store.name_of(entity);
    if let Some(&pos) = store.get::<Position>(entity) {
        elog!("removed {pos:?} from {name:?}");
    }
    store.despawn(entity).unwrap();
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let entity = store.entity_named("Bob");

    set_position_with_hooks(&mut store, entity, Position { x: 10.0, y: 20.0 });

    // This operation changes the entity's archetype in Store, which invokes a row move
    store.set(entity, EntityTag).unwrap();

    destruct_with_hooks(&mut store, entity);
}

#[test]
fn test_entity_hooks() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "added Position to \"Bob\"",
            "set Position { x: 10.0, y: 20.0 } for \"Bob\"",
            "removed Position { x: 10.0, y: 20.0 } from \"Bob\"",
        ]
    );
}
