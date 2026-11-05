//! Port of `examples/flecs/reflection/reflection_ser_std_string.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StringComponent {
    pub a: String,
    pub b: String,
}

impl StringComponent {
    pub fn to_flecs_json(&self) -> String {
        let v = serde_json::to_value(self).unwrap();
        format!("{{\"a\": {}, \"b\": {}}}", v["a"], v["b"])
    }

    pub fn to_flecs_expr(&self) -> String {
        format!("{{a: {:?}, b: {:?}}}", self.a, self.b)
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity & serialize StringComponent
    let e = store.spawn_auto().unwrap();
    store
        .set(
            e,
            StringComponent {
                a: "foo".to_string(),
                b: "bar".to_string(),
            },
        )
        .unwrap();

    if let Some(v) = store.get::<StringComponent>(e) {
        elog!("{}", v.to_flecs_json());
    }

    // Deserialize new strings into value in Store
    store
        .modify::<StringComponent, _>(e, |v| {
            *v = serde_json::from_str("{\"a\": \"hello\", \"b\": \"world\"}").unwrap();
        })
        .unwrap();

    if let Some(v) = store.get::<StringComponent>(e) {
        elog!("{}", v.to_flecs_expr());
    }
}

#[test]
fn test_reflection_ser_std_string() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "{\"a\": \"foo\", \"b\": \"bar\"}",
            "{a: \"hello\", b: \"world\"}",
        ]
    );
}
