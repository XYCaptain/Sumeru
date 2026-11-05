//! Port of `examples/flecs/systems/system_sync_point_delete.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create test entities for a (Position, Velocity) query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 0.0, y: 0.0 }).unwrap();
    store.set(e1, Velocity { x: 1.0, y: 2.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 1.0, y: 2.0 }).unwrap();
    store.set(e2, Velocity { x: 1.0, y: 2.0 }).unwrap();

    loop {
        // System 1 (Move): in-place update of Position by Velocity
        {
            let mut q = store.query_mut::<(&mut Position, &Velocity)>().unwrap();
            for (_e, (p, v)) in q.iter() {
                p.x += v.x;
                p.y += v.y;
            }
        }

        // System 2 (DeleteEntity): enqueue delete when p.x >= 3.0
        let mut cmd = store.commands();
        for (e, p) in store.query::<&Position>().iter() {
            if p.x >= 3.0 {
                elog!("Delete entity {}", store.name_of(e));
                cmd.delete_entity(e);
            }
        }

        // Sync point: merge deferred deletes before PrintPosition runs
        cmd.apply_discard(&mut store).unwrap();

        // System 3 (PrintPosition): never prints entities deleted by DeleteEntity
        let mut remaining = 0;
        for (e, p) in store.query::<&Position>().iter() {
            elog!("{}: {{ {}, {} }}", store.name_of(e), p.x, p.y);
            remaining += 1;
        }

        if remaining == 0 {
            break;
        }
    }
}

#[test]
fn test_system_sync_point_delete() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "e1: { 1, 2 }",
            "e2: { 2, 4 }",
            "Delete entity e2",
            "e1: { 2, 4 }",
            "Delete entity e1",
        ]
    );
}
