//! Port of `examples/flecs/entities/component_attributes.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;
use sumeru::SchemaVersion;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct WorldPosition;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Eats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Apples;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Likes {
    pub score: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Bob;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Register Position schema and component metadata attributes in Store
    store
        .register_components([ComponentRegistration::of_unsupported::<Position>()])
        .unwrap();
    let _schema = ComponentSchema::of::<Position>(SchemaVersion::new(1));

    let apples = store.entity_named("Apples");
    let bob_tag = store.entity_named("BobTag");
    let pos_meta = store.entity_named("Position");
    store.set(pos_meta, WorldPosition).unwrap();
    store.set_pair(pos_meta, Eats, apples).unwrap();
    store.set_pair(pos_meta, Likes { score: 10 }, bob_tag).unwrap();

    let e = store.entity_named("Bob");

    // Set Position on Bob (triggers on_add + on_set hooks)
    let was_present = store.has::<Position>(e);
    let new_pos = Position { x: 1.0, y: 2.0 };
    store.set(e, new_pos).unwrap();
    if !was_present {
        elog!("added Position to {}", store.name_of(e));
    }
    if let Some(p) = store.get::<Position>(e) {
        elog!("set {:?} for {}", p, store.name_of(e));
    }

    // Remove Position from Bob (triggers on_remove hook)
    if let Some(removed) = store.remove::<Position>(e).unwrap() {
        elog!("removed {:?} from {}", removed, store.name_of(e));
    }
}

#[test]
fn test_component_attributes() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "added Position to Bob",
            "set Position { x: 1.0, y: 2.0 } for Bob",
            "removed Position { x: 1.0, y: 2.0 } from Bob",
        ]
    );
}
