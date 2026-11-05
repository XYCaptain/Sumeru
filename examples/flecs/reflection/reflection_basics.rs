//! Port of `examples/flecs/reflection/reflection_basics.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;
use sumer_state::SchemaVersion;

pub struct PosXField;
pub struct PosYField;

impl ComponentField<Position> for PosXField {
    type Value = f32;

    fn extract<'a>(component: &'a Position) -> &'a Self::Value {
        &component.x
    }

    fn field_name() -> &'static str {
        "x"
    }
}

impl ComponentField<Position> for PosYField {
    type Value = f32;

    fn extract<'a>(component: &'a Position) -> &'a Self::Value {
        &component.y
    }

    fn field_name() -> &'static str {
        "y"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_unsupported::<Position>()])
        .unwrap();
    let _schema = ComponentSchema::of::<Position>(SchemaVersion::new(1));

    // Create a new entity
    let e = store.spawn_auto().unwrap();
    store.set(e, Position { x: 2.0, y: 4.0 }).unwrap();

    // Convert position component to flecs expression string
    if let Some(p) = store.get::<Position>(e) {
        let expr = format!(
            "{{{}: {}, {}: {}}}",
            PosXField::field_name(),
            PosXField::extract(p),
            PosYField::field_name(),
            PosYField::extract(p)
        );
        elog!("Position: {expr}");
    }
}

#[test]
fn test_reflection_basics() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["Position: {x: 2, y: 4}"]);
}
