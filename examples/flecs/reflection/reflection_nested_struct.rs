//! Port of `examples/flecs/reflection/reflection_nested_struct.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;
use sumeru::SchemaVersion;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Line {
    pub start: Point,
    pub stop: Point,
}

pub struct StartField;
pub struct StopField;
pub struct PointXField;
pub struct PointYField;

impl ComponentField<Line> for StartField {
    type Value = Point;

    fn extract<'a>(component: &'a Line) -> &'a Self::Value {
        &component.start
    }

    fn field_name() -> &'static str {
        "start"
    }
}

impl ComponentField<Line> for StopField {
    type Value = Point;

    fn extract<'a>(component: &'a Line) -> &'a Self::Value {
        &component.stop
    }

    fn field_name() -> &'static str {
        "stop"
    }
}

impl ComponentField<Point> for PointXField {
    type Value = f32;

    fn extract<'a>(component: &'a Point) -> &'a Self::Value {
        &component.x
    }

    fn field_name() -> &'static str {
        "x"
    }
}

impl ComponentField<Point> for PointYField {
    type Value = f32;

    fn extract<'a>(component: &'a Point) -> &'a Self::Value {
        &component.y
    }

    fn field_name() -> &'static str {
        "y"
    }
}

fn point_to_expr(p: &Point) -> String {
    format!(
        "{{{}: {}, {}: {}}}",
        PointXField::field_name(),
        PointXField::extract(p),
        PointYField::field_name(),
        PointYField::extract(p)
    )
}

fn line_to_expr(line: &Line) -> String {
    format!(
        "{{{}: {}, {}: {}}}",
        StartField::field_name(),
        point_to_expr(StartField::extract(line)),
        StopField::field_name(),
        point_to_expr(StopField::extract(line))
    )
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    store
        .register_components([
            ComponentRegistration::of_unsupported::<Point>(),
            ComponentRegistration::of_unsupported::<Line>(),
        ])
        .unwrap();
    let _point_schema = ComponentSchema::of::<Point>(SchemaVersion::new(1));
    let _line_schema = ComponentSchema::of::<Line>(SchemaVersion::new(1));

    // Create entity, set Line component
    let e = store.spawn_auto().unwrap();
    store
        .set(
            e,
            Line {
                start: Point { x: 10.0, y: 20.0 },
                stop: Point { x: 30.0, y: 40.0 },
            },
        )
        .unwrap();

    // Convert Line component to flecs expression string
    if let Some(line) = store.get::<Line>(e) {
        elog!("{}", line_to_expr(line));
    }
}

#[test]
fn test_reflection_nested_struct() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["{start: {x: 10, y: 20}, stop: {x: 30, y: 40}}"]);
}
