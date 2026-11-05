//! Port of `examples/flecs/systems/system_pipeline.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create test entities for a (Position, Velocity) query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e1, Velocity { x: 1.0, y: 2.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e2, Velocity { x: 3.0, y: 4.0 }).unwrap();

    // Phase 1 (OnUpdate): Move system updates Position by Velocity
    {
        let mut q = store.query_mut::<(&mut Position, &Velocity)>().unwrap();
        for (_e, (p, v)) in q.iter() {
            p.x += v.x;
            p.y += v.y;
        }
    }

    // Phase 2 (PostUpdate): Print system logs resulting Position
    for (e, p) in store.query::<&Position>().iter() {
        elog!("{}: {{ {}, {} }}", store.name_of(e), p.x, p.y);
    }
}

#[test]
fn test_system_pipeline() {
    main();
    assert_eq!(
        take_log(),
        vec!["e1: { 11, 22 }", "e2: { 13, 24 }"]
    );
}
