//! Port of `examples/flecs/reflection/reflection_runtime_component.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeComponentSchema {
    pub name: String,
    pub members: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RuntimeComponentData {
    pub values: Vec<f32>,
}

pub struct RuntimeCursor<'a> {
    data: &'a mut RuntimeComponentData,
    index: usize,
}

impl<'a> RuntimeCursor<'a> {
    pub fn new(data: &'a mut RuntimeComponentData, member_count: usize) -> Self {
        if data.values.len() < member_count {
            data.values.resize(member_count, 0.0);
        }
        Self { data, index: 0 }
    }

    pub fn push(&mut self) {
        self.index = 0;
    }

    pub fn next(&mut self) {
        self.index += 1;
    }

    pub fn set_float(&mut self, val: f32) {
        self.data.values[self.index] = val;
    }

    pub fn pop(&mut self) {}
}

pub fn to_expr_id(schema: &RuntimeComponentSchema, data: &RuntimeComponentData) -> String {
    let pairs: Vec<String> = schema
        .members
        .iter()
        .zip(data.values.iter())
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    format!("{{{}}}", pairs.join(", "))
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Register runtime-described "Position" component schema as an entity in Store
    let position_schema_entity = store.entity_named("Position");
    let schema = RuntimeComponentSchema {
        name: "Position".to_string(),
        members: vec!["x".to_string(), "y".to_string()],
    };
    store.set(position_schema_entity, schema).unwrap();

    // Create entity and attach runtime component data
    let e = store.spawn_auto().unwrap();
    store.set(e, RuntimeComponentData::default()).unwrap();

    let member_count = store
        .get::<RuntimeComponentSchema>(position_schema_entity)
        .unwrap()
        .members
        .len();

    // Set value of position using reflection cursor API
    store
        .modify::<RuntimeComponentData, _>(e, |data| {
            let mut cur = RuntimeCursor::new(data, member_count);
            cur.push();
            cur.set_float(10.0);
            cur.next();
            cur.set_float(20.0);
            cur.pop();
        })
        .unwrap();

    let schema_ref = store
        .get::<RuntimeComponentSchema>(position_schema_entity)
        .unwrap();
    let data_ref = store.get::<RuntimeComponentData>(e).unwrap();
    elog!("{}", to_expr_id(schema_ref, data_ref));
}

#[test]
fn test_reflection_runtime_component() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["{x: 10, y: 20}"]);
}
