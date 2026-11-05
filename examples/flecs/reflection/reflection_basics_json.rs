//! Port of `examples/flecs/reflection/reflection_basics_json.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

fn position_to_json(p: &Position) -> String {
    let val = serde_json::json!({
        "x": p.x as i64,
        "y": p.y as i64,
    });
    format!("{{\"x\":{}, \"y\":{}}}", val["x"], val["y"])
}

fn entity_to_json(store: &Store, e: EntityId) -> String {
    let name = store.name_of(e);
    let pos_json = store
        .get::<Position>(e)
        .map(position_to_json)
        .unwrap_or_else(|| "null".to_string());
    format!("{{\"name\":\"{name}\", \"components\":{{\"Position\":{pos_json}}}}}")
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a new entity with the Position component
    let e = store.spawn_auto().unwrap();
    store.set(e, Position { x: 2.0, y: 4.0 }).unwrap();

    // Convert position component to JSON string
    if let Some(p) = store.get::<Position>(e) {
        let expr = position_to_json(p);
        elog!("Position: {expr}");
    }

    // Convert entity to JSON string
    elog!("Entity: {}", entity_to_json(&store, e));
}

#[test]
fn test_reflection_basics_json() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Position: {\"x\":2, \"y\":4}",
            "Entity: {\"name\":\"#1\", \"components\":{\"Position\":{\"x\":2, \"y\":4}}}",
        ]
    );
}
