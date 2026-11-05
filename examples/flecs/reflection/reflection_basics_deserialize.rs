//! Port of `examples/flecs/reflection/reflection_basics_deserialize.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

pub struct PositionCursor<'a> {
    target: &'a mut Position,
    field_index: usize,
}

impl<'a> PositionCursor<'a> {
    pub fn new(target: &'a mut Position) -> Self {
        Self {
            target,
            field_index: 0,
        }
    }

    pub fn push(&mut self) {
        self.field_index = 0;
    }

    pub fn next(&mut self) {
        self.field_index += 1;
    }

    pub fn member(&mut self, name: &str) {
        self.field_index = match name {
            "x" => 0,
            "y" => 1,
            _ => panic!("unknown member: {name}"),
        };
    }

    pub fn set_float(&mut self, value: f32) {
        match self.field_index {
            0 => self.target.x = value,
            1 => self.target.y = value,
            _ => panic!("field index out of bounds"),
        }
    }

    pub fn pop(&mut self) {}
}

fn to_expr(pos: &Position) -> String {
    format!("{{x: {}, y: {}}}", pos.x, pos.y)
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a new entity, set value of position using reflection cursor API
    let e = store.spawn_auto().unwrap();
    store.set(e, Position::default()).unwrap();

    store
        .modify::<Position, _>(e, |pos| {
            let mut cur = PositionCursor::new(pos);
            cur.push(); // {
            cur.set_float(10.0); //   10
            cur.next(); //   ,
            cur.set_float(20.0); //   20
            cur.pop(); // }

            elog!("{}", to_expr(pos));
        })
        .unwrap();

    // Use member names before assigning values
    store
        .modify::<Position, _>(e, |pos| {
            let mut cur = PositionCursor::new(pos);
            cur.push(); // {
            cur.member("y"); //   y:
            cur.set_float(10.0); //   10
            cur.member("x"); //   x:
            cur.set_float(20.0); //   20
            cur.pop(); // }

            elog!("{}", to_expr(pos));
        })
        .unwrap();
}

#[test]
fn test_reflection_basics_deserialize() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["{x: 10, y: 20}", "{x: 20, y: 10}"]);
}
