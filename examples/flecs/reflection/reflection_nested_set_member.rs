//! Port of `examples/flecs/reflection/reflection_nested_set_member.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

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

#[derive(Clone, Copy)]
enum ActivePoint {
    Start,
    Stop,
}

#[derive(Clone, Copy)]
enum ActiveCoord {
    X,
    Y,
}

pub struct LineCursor<'a> {
    target: &'a mut Line,
    depth: usize,
    point: ActivePoint,
    coord: ActiveCoord,
}

impl<'a> LineCursor<'a> {
    pub fn new(target: &'a mut Line) -> Self {
        Self {
            target,
            depth: 0,
            point: ActivePoint::Start,
            coord: ActiveCoord::X,
        }
    }

    pub fn push(&mut self) {
        self.depth += 1;
    }

    pub fn pop(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    pub fn member(&mut self, name: &str) {
        if self.depth == 1 {
            self.point = match name {
                "start" => ActivePoint::Start,
                "stop" => ActivePoint::Stop,
                _ => panic!("unknown Line member: {name}"),
            };
        } else if self.depth == 2 {
            self.coord = match name {
                "x" => ActiveCoord::X,
                "y" => ActiveCoord::Y,
                _ => panic!("unknown Point member: {name}"),
            };
        }
    }

    pub fn set_float(&mut self, val: f32) {
        let pt = match self.point {
            ActivePoint::Start => &mut self.target.start,
            ActivePoint::Stop => &mut self.target.stop,
        };
        match self.coord {
            ActiveCoord::X => pt.x = val,
            ActiveCoord::Y => pt.y = val,
        }
    }
}

fn line_to_expr(line: &Line) -> String {
    format!(
        "{{start: {{x: {}, y: {}}}, stop: {{x: {}, y: {}}}}}",
        line.start.x, line.start.y, line.stop.x, line.stop.y
    )
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity, set value of Line using reflection cursor API
    let e = store.spawn_auto().unwrap();
    store.set(e, Line::default()).unwrap();

    store
        .modify::<Line, _>(e, |line| {
            let mut cur = LineCursor::new(line);

            cur.push(); // {
            cur.member("start"); //   start:
            cur.push(); //   {
            cur.member("x"); //     x:
            cur.set_float(10.0); //     10
            cur.member("y"); //     y:
            cur.set_float(20.0); //     20
            cur.pop(); //   }
            cur.member("stop"); //   stop:
            cur.push(); //   {
            cur.member("x"); //     x:
            cur.set_float(30.0); //     30
            cur.member("y"); //     y:
            cur.set_float(40.0); //     40
            cur.pop(); //   }
            cur.pop(); // }

            // Convert component to string
            elog!("{}", line_to_expr(line));
        })
        .unwrap();
}

#[test]
fn test_reflection_nested_set_member() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["{start: {x: 10, y: 20}, stop: {x: 30, y: 40}}"]);
}
