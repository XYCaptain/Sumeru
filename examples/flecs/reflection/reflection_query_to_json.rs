//! Port of `examples/flecs/reflection/reflection_query_to_json.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Mass {
    pub value: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let a = store.entity_named("a");
    store.set(a, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(a, Velocity { x: 1.0, y: 2.0 }).unwrap();

    let b = store.entity_named("b");
    store.set(b, Position { x: 20.0, y: 30.0 }).unwrap();
    store.set(b, Velocity { x: 2.0, y: 3.0 }).unwrap();

    let c = store.entity_named("c");
    store.set(c, Position { x: 30.0, y: 40.0 }).unwrap();
    store.set(c, Velocity { x: 3.0, y: 4.0 }).unwrap();
    store.set(c, Mass { value: 10.0 }).unwrap();

    let d = store.entity_named("d");
    store.set(d, Position { x: 30.0, y: 40.0 }).unwrap();
    store.set(d, Velocity { x: 4.0, y: 5.0 }).unwrap();
    store.set(d, Mass { value: 20.0 }).unwrap();

    // Query for components on Store
    let mut rows: Vec<(String, Position, Velocity)> = store
        .query::<(&Position, &Velocity)>()
        .iter()
        .map(|(entity, (pos, vel))| (store.name_of(entity), *pos, *vel))
        .collect();
    rows.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let results: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|(name, pos, vel)| {
            serde_json::json!({
                "name": name,
                "fields": {
                    "values": [
                        { "x": pos.x as i64, "y": pos.y as i64 },
                        { "x": vel.x as i64, "y": vel.y as i64 }
                    ]
                }
            })
        })
        .collect();

    let payload = serde_json::json!({ "results": results });
    elog!("{}", serde_json::to_string(&payload).unwrap());
}

#[test]
fn test_reflection_query_to_json() {
    main();
    let lines = take_log();
    assert_eq!(lines.len(), 1);
    let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    let results = parsed["results"].as_array().unwrap();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0]["name"], "a");
    assert_eq!(results[0]["fields"]["values"][0]["y"], 20);
    assert_eq!(results[1]["name"], "b");
    assert_eq!(results[2]["name"], "c");
    assert_eq!(results[3]["name"], "d");
    assert_eq!(results[3]["fields"]["values"][1]["x"], 4);
}
