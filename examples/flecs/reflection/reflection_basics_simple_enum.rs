//! Port of `examples/flecs/reflection/reflection_basics_simple_enum.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;
use sumer_state::SchemaVersion;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeWithEnum {
    pub color: Color,
}

pub struct ColorField;

impl ComponentField<TypeWithEnum> for ColorField {
    type Value = Color;

    fn extract<'a>(component: &'a TypeWithEnum) -> &'a Self::Value {
        &component.color
    }

    fn field_name() -> &'static str {
        "color"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_unsupported::<TypeWithEnum>()])
        .unwrap();
    let _schema = ComponentSchema::of::<TypeWithEnum>(SchemaVersion::new(1));

    // Create a new entity
    let e = store.spawn_auto().unwrap();
    store
        .set(
            e,
            TypeWithEnum {
                color: Color::Green,
            },
        )
        .unwrap();

    // Convert TypeWithEnum component to flecs expression string
    if let Some(p) = store.get::<TypeWithEnum>(e) {
        let expr = format!("{{{}: {:?}}}", ColorField::field_name(), ColorField::extract(p));
        elog!("TypeWithEnum: {expr}");
    }
}

#[test]
fn test_reflection_basics_simple_enum() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["TypeWithEnum: {color: Green}"]);
}
