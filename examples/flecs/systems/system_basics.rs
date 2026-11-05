//! Port of `examples/flecs/systems/system_basics.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a few test entities for a (Position, Velocity) query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e1, Velocity { x: 1.0, y: 2.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e2, Velocity { x: 3.0, y: 4.0 }).unwrap();

    // This entity will not match as it does not have both Position and Velocity
    let e3 = store.entity_named("e3");
    store.set(e3, Position { x: 10.0, y: 20.0 }).unwrap();

    // Run the system over (&mut Position, &Velocity)
    let mut updated = Vec::new();
    {
        let mut q = store.query_mut::<(&mut Position, &Velocity)>().unwrap();
        for (e, (p, v)) in q.iter() {
            p.x += v.x;
            p.y += v.y;
            updated.push((e, p.x, p.y));
        }
    }

    for (e, x, y) in updated {
        elog!("{}: {{ {}, {} }}", store.name_of(e), x, y);
    }
}

#[test]
fn test_system_basics() {
    main();
    assert_eq!(
        take_log(),
        vec!["e1: { 11, 22 }", "e2: { 13, 24 }"]
    );
}
