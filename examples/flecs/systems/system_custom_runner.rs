//! Port of `examples/flecs/systems/system_custom_runner.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

fn run_move_system(
    store: &mut Store,
    mut each: impl FnMut(&Store, EntityId, &Position, &Velocity),
) {
    elog!("Move begin");

    let mut snapshot = Vec::new();
    {
        let mut q = store.query_mut::<(&mut Position, &Velocity)>().unwrap();
        for (e, (pos, vel)) in q.iter() {
            pos.x += vel.x;
            pos.y += vel.y;
            snapshot.push((e, *pos, *vel));
        }
    }

    for (e, pos, vel) in &snapshot {
        each(store, *e, pos, vel);
    }

    elog!("Move end");
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a few test entities for a Position, Velocity query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e1, Velocity { x: 1.0, y: 2.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(e2, Velocity { x: 3.0, y: 4.0 }).unwrap();

    // This entity will not match as it does not have Position, Velocity
    let e3 = store.entity_named("e3");
    store.set(e3, Position { x: 10.0, y: 20.0 }).unwrap();

    // Run the system with custom runner
    run_move_system(&mut store, |s, e, pos, _vel| {
        elog!("{}: {{ {}, {} }}", s.name_of(e), pos.x, pos.y);
    });
}

#[test]
fn test_system_custom_runner() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Move begin",
            "e1: { 11, 22 }",
            "e2: { 13, 24 }",
            "Move end",
        ]
    );
}
