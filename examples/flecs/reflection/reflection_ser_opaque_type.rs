//! Port of `examples/flecs/reflection/reflection_ser_opaque_type.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

// Use opaque reflection support to add a computed `result` member to type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Sum {
    pub a: i32,
    pub b: i32,
}

impl Sum {
    pub fn to_opaque_json(&self) -> String {
        let result = self.a + self.b;
        format!("{{\"a\":{}, \"b\":{}, \"result\":{}}}", self.a, self.b, result)
    }

    /// Mirrors upstream Flecs `ensure_member` behavior (`if member != "a" -> &mut dst.a else if member != "b" -> &mut dst.b`),
    /// while ignoring computed `"result"`.
    pub fn from_opaque_json(&mut self, json_str: &str) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        if let Some(obj) = parsed.as_object() {
            for (member, val) in obj {
                if let Some(v) = val.as_i64() {
                    if member != "a" {
                        self.a = v as i32;
                    } else if member != "b" {
                        self.b = v as i32;
                    }
                }
            }
        }
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let e = store.spawn_auto().unwrap();
    store.set(e, Sum { a: 10, b: 20 }).unwrap();

    // Serialize value of Sum to JSON
    if let Some(v) = store.get::<Sum>(e) {
        elog!("{}", v.to_opaque_json());
    }

    // Deserialize new value into Sum on Store
    store
        .modify::<Sum, _>(e, |v| {
            v.from_opaque_json("{\"a\": 20, \"b\": 22}");
        })
        .unwrap();

    // Serialize value again
    if let Some(v) = store.get::<Sum>(e) {
        elog!("{}", v.to_opaque_json());
    }
}

#[test]
fn test_reflection_ser_opaque_type() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "{\"a\":10, \"b\":20, \"result\":30}",
            "{\"a\":22, \"b\":20, \"result\":42}",
        ]
    );
}
