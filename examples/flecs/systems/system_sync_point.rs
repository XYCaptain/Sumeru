//! Port of `examples/flecs/systems/system_sync_point.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PositionSP {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct VelocitySP {
    pub x: f32,
    pub y: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create test entities for a (PositionSP, VelocitySP) query
    let e1 = store.entity_named("e1");
    store.set(e1, PositionSP { x: 10.0, y: 20.0 }).unwrap();
    store.set(e1, VelocitySP { x: 1.0, y: 2.0 }).unwrap();

    let e2 = store.entity_named("e2");
    store.set(e2, PositionSP { x: 10.0, y: 20.0 }).unwrap();
    store.set(e2, VelocitySP { x: 3.0, y: 4.0 }).unwrap();

    // System 1 (SetVelocitySP): deferred `set` of VelocitySP for entities with PositionSP
    let mut cmd = store.commands();
    for (e, _pos) in store.query::<&PositionSP>().iter() {
        cmd.entity(e).set(VelocitySP { x: 1.0, y: 2.0 });
    }

    // Sync point: merge deferred commands before Move reads VelocitySP
    cmd.apply_discard(&mut store).unwrap();

    // System 2 (Move): reads VelocitySP and updates PositionSP in-place
    {
        let mut q = store.query_mut::<(&mut PositionSP, &VelocitySP)>().unwrap();
        for (_e, (p, v)) in q.iter() {
            p.x += v.x;
            p.y += v.y;
        }
    }

    // System 3 (PrintPositionSP): prints resulting PositionSP
    for (e, p) in store.query::<&PositionSP>().iter() {
        elog!("{}: {{ {}, {} }}", store.name_of(e), p.x, p.y);
    }
}

#[test]
fn test_system_sync_point() {
    main();
    assert_eq!(
        take_log(),
        vec!["e1: { 11, 22 }", "e2: { 11, 22 }"]
    );
}
