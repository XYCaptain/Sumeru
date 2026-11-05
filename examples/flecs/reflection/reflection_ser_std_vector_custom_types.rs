//! Port of `examples/flecs/reflection/reflection_ser_std_vector_custom_types.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct VectorComponent {
    pub points: Vec<Point>,
    pub strings: Vec<String>,
}

impl VectorComponent {
    pub fn to_flecs_json(&self) -> String {
        let points_str = self
            .points
            .iter()
            .map(|p| format!("{{\"x\":{}, \"y\":{}}}", p.x as i64, p.y as i64))
            .collect::<Vec<_>>()
            .join(", ");
        let strings_str = self
            .strings
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{\"points\":[{points_str}], \"strings\":[{strings_str}]}}")
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create value in Store & serialize it to JSON
    let e = store.spawn_auto().unwrap();
    store
        .set(
            e,
            VectorComponent {
                points: vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }],
                strings: vec!["foo".to_string(), "bar".to_string()],
            },
        )
        .unwrap();

    if let Some(v) = store.get::<VectorComponent>(e) {
        elog!("{}", v.to_flecs_json());
    }

    // Deserialize new values from JSON into Store component
    store
        .modify::<VectorComponent, _>(e, |v| {
            *v = serde_json::from_str(
                "{\"points\": [{\"x\": 4.0, \"y\": 5.0}, {\"x\": 6.0, \"y\": 7.0}], \"strings\":[\"hello\", \"flecs\", \"reflection\"]}",
            )
            .unwrap();
        })
        .unwrap();

    // Serialize again
    if let Some(v) = store.get::<VectorComponent>(e) {
        elog!("{}", v.to_flecs_json());
    }
}

#[test]
fn test_reflection_ser_std_vector_custom_types() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "{\"points\":[{\"x\":1, \"y\":2}, {\"x\":3, \"y\":4}], \"strings\":[\"foo\", \"bar\"]}",
            "{\"points\":[{\"x\":4, \"y\":5}, {\"x\":6, \"y\":7}], \"strings\":[\"hello\", \"flecs\", \"reflection\"]}",
        ]
    );
}
