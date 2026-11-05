//! Port of `examples/flecs/reflection/reflection_runtime_nested_component.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStructSchema {
    pub name: String,
    pub members: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RuntimeNestedData {
    pub slots: Vec<[f32; 2]>,
}

pub struct NestedCursor<'a> {
    data: &'a mut RuntimeNestedData,
    depth: usize,
    outer_idx: usize,
    inner_idx: usize,
}

impl<'a> NestedCursor<'a> {
    pub fn new(data: &'a mut RuntimeNestedData, outer_len: usize) -> Self {
        if data.slots.len() < outer_len {
            data.slots.resize(outer_len, [0.0, 0.0]);
        }
        Self {
            data,
            depth: 0,
            outer_idx: 0,
            inner_idx: 0,
        }
    }

    pub fn push(&mut self) {
        self.depth += 1;
        if self.depth == 1 {
            self.outer_idx = 0;
        } else if self.depth == 2 {
            self.inner_idx = 0;
        }
    }

    pub fn next(&mut self) {
        if self.depth == 1 {
            self.outer_idx += 1;
        } else if self.depth == 2 {
            self.inner_idx += 1;
        }
    }

    pub fn set_float(&mut self, val: f32) {
        self.data.slots[self.outer_idx][self.inner_idx] = val;
    }

    pub fn pop(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

pub fn to_nested_expr(
    outer: &RuntimeStructSchema,
    inner: &RuntimeStructSchema,
    data: &RuntimeNestedData,
) -> String {
    let sections: Vec<String> = outer
        .members
        .iter()
        .zip(data.slots.iter())
        .map(|(outer_member, vals)| {
            let inner_pairs: Vec<String> = inner
                .members
                .iter()
                .zip(vals.iter())
                .map(|(k, v)| format!("{k}: {v:.2}"))
                .collect();
            format!("{outer_member}: {{{}}}", inner_pairs.join(", "))
        })
        .collect();
    format!("{{{}}}", sections.join(", "))
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let point = store.entity_named("Point");
    store
        .set(
            point,
            RuntimeStructSchema {
                name: "Point".to_string(),
                members: vec!["x".to_string(), "y".to_string()],
            },
        )
        .unwrap();

    let line = store.entity_named("Line");
    store
        .set(
            line,
            RuntimeStructSchema {
                name: "Line".to_string(),
                members: vec!["start".to_string(), "stop".to_string()],
            },
        )
        .unwrap();

    // Create entity, set value of line using reflection cursor API
    let e = store.spawn_auto().unwrap();
    store.set(e, RuntimeNestedData::default()).unwrap();

    let outer_len = store
        .get::<RuntimeStructSchema>(line)
        .unwrap()
        .members
        .len();

    #[rustfmt::skip]
    fn cursor(cur: &mut NestedCursor<'_>) {
        cur.push();            // {
        cur.push();            //   {
        cur.set_float(10.0);   //     10
        cur.next();            //     ,
        cur.set_float(20.0);   //     20
        cur.pop();             //   }
        cur.next();            //   ,
        cur.push();            //   {
        cur.set_float(30.0);   //     30
        cur.next();            //     ,
        cur.set_float(40.0);   //     40
        cur.pop();             //   }
        cur.pop();             // }
    }

    store
        .modify::<RuntimeNestedData, _>(e, |data| {
            let mut cur = NestedCursor::new(data, outer_len);
            cursor(&mut cur);
        })
        .unwrap();

    let line_schema = store.get::<RuntimeStructSchema>(line).unwrap();
    let point_schema = store.get::<RuntimeStructSchema>(point).unwrap();
    let data = store.get::<RuntimeNestedData>(e).unwrap();
    elog!("{}", to_nested_expr(line_schema, point_schema, data));
}

#[test]
fn test_reflection_runtime_nested_component() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec!["{start: {x: 10.00, y: 20.00}, stop: {x: 30.00, y: 40.00}}"]
    );
}
