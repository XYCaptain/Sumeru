//! Port of `examples/flecs/reflection/reflection_ser_std_vector_builtin_types.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VectorComponent {
    pub ints: Vec<i32>,
    pub strings: Vec<String>,
}

impl VectorComponent {
    pub fn to_flecs_json(&self) -> String {
        let ints_str = self
            .ints
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let strings_str = self
            .strings
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{\"ints\":[{ints_str}], \"strings\":[{strings_str}]}}")
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
                ints: vec![1, 2, 3],
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
                "{\"ints\": [4, 5], \"strings\":[\"hello\", \"flecs\", \"reflection\"]}",
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
fn test_reflection_ser_std_vector_builtin_types() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "{\"ints\":[1, 2, 3], \"strings\":[\"foo\", \"bar\"]}",
            "{\"ints\":[4, 5], \"strings\":[\"hello\", \"flecs\", \"reflection\"]}",
        ]
    );
}
