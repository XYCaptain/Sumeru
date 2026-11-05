//! Port of `examples/flecs/systems/system_ctx.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Radius {
    pub value: f32,
}

fn sqr(value: f32) -> f32 {
    value * value
}

fn distance_sqr(p1: &Position, p2: &Position) -> f32 {
    sqr(p2.x - p1.x) + sqr(p2.y - p1.y)
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Deterministic pseudo-random coordinates and radii for 10 entities so collision
    // detection via a secondary context query is 100% reproducible.
    let configs = [
        (10.0, 10.0, 6.0),
        (14.0, 12.0, 5.0),
        (40.0, 40.0, 4.0),
        (12.0, 9.0, 4.0),
        (42.0, 39.0, 5.0),
        (25.0, 25.0, 3.0),
        (26.0, 24.0, 3.0),
        (5.0, 45.0, 2.0),
        (27.0, 25.0, 4.0),
        (11.0, 11.0, 5.0),
    ];

    for (x, y, r) in configs {
        let e = store.spawn_auto().unwrap();
        store.set(e, Position { x, y }).unwrap();
        store.set(e, Radius { value: r }).unwrap();
    }

    // Build a reusable query object passed as system context
    let query_collide = Query::<(&Position, &Radius)>::from_store(&store).unwrap();

    // System iterates primary query and uses context query for pairwise collision checks
    for (e1, (p1, r1)) in store.query::<(&Position, &Radius)>().iter() {
        query_collide
            .for_each(&store, |(e2, (p2, r2))| {
                if e1 == e2 {
                    return;
                }
                if e1 > e2 {
                    return;
                }

                let d_sqr = distance_sqr(p1, p2);
                let r_sqr = sqr(r1.value + r2.value);
                if r_sqr > d_sqr {
                    elog!(
                        "{} and {} collided!",
                        EntityHandle::index(e1),
                        EntityHandle::index(e2)
                    );
                }
            })
            .unwrap();
    }
}

#[test]
fn test_system_ctx() {
    main();
    let lines = take_log();
    assert!(!lines.is_empty());
    assert!(lines.iter().all(|l| l.ends_with("collided!")));
}
