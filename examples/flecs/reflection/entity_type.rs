//! Port of `examples/flecs/reflection/entity_type.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;
use sumer_state::SchemaVersion;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeWithEntity {
    pub e: EntityId,
}

pub struct EntityMemberField;

impl ComponentField<TypeWithEntity> for EntityMemberField {
    type Value = EntityId;

    fn extract<'a>(component: &'a TypeWithEntity) -> &'a Self::Value {
        &component.e
    }

    fn field_name() -> &'static str {
        "e"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_unsupported::<TypeWithEntity>()])
        .unwrap();
    let _schema = ComponentSchema::of::<TypeWithEntity>(SchemaVersion::new(1));

    let bar = store.entity_named("bar");

    // Create a new entity with the TypeWithEntity component
    let e = store.spawn_auto().unwrap();
    store.set(e, TypeWithEntity { e: bar }).unwrap();

    // Convert TypeWithEntity component to expression string via ComponentField reflection
    if let Some(p) = store.get::<TypeWithEntity>(e) {
        let target = *EntityMemberField::extract(p);
        let expr = format!("{{{}: {}}}", EntityMemberField::field_name(), store.name_of(target));
        elog!("TypeWithEntity: {expr}");
    }
}

#[test]
fn test_entity_type() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["TypeWithEntity: {e: bar}"]);
}
