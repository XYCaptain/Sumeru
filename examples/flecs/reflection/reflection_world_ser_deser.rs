//! Port of `examples/flecs/reflection/reflection_world_ser_deser.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct SimName(pub String);

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Default,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct SimPosition {
    pub x: f32,
    pub y: f32,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Default,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct SimVelocity {
    pub x: f32,
    pub y: f32,
}

fn import_move_module(store: &mut Store) {
    store
        .register_components([
            ComponentRegistration::of::<SimName>(),
            ComponentRegistration::of::<SimPosition>(),
            ComponentRegistration::of::<SimVelocity>(),
        ])
        .unwrap();
}

fn run_move_system(store: &mut Store) {
    let mut moved: Vec<(String, f32, f32)> = Vec::new();
    {
        let mut q = store
            .query_mut::<(&SimName, &mut SimPosition, &SimVelocity)>()
            .unwrap();
        for (_entity, (name, pos, vel)) in q.iter() {
            pos.x += vel.x;
            pos.y += vel.y;
            moved.push((format!("::{}", name.0), pos.x, pos.y));
        }
    }
    moved.sort_by(|a, b| a.0.cmp(&b.0));
    for (path, x, y) in moved {
        elog!("{path} moved to {{x: {x}, y: {y}}}");
    }
}

fn store_to_json_world(store: &Store) -> String {
    let mut rows: Vec<(String, u32, SimPosition, SimVelocity)> = store
        .query::<(&SimName, &SimPosition, &SimVelocity)>()
        .iter()
        .map(|(e, (n, p, v))| (n.0.clone(), EntityHandle::index(e), *p, *v))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let entities_json: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|(name, id, pos, vel)| {
            serde_json::json!({
                "name": name,
                "id": id,
                "components": {
                    "Position": { "x": pos.x as i64, "y": pos.y as i64 },
                    "Velocity": { "x": vel.x as i64, "y": vel.y as i64 }
                }
            })
        })
        .collect();
    serde_json::to_string(&serde_json::json!({ "results": entities_json })).unwrap()
}

pub fn main() {
    clear_log();
    let mut world_a = Store::new();
    import_move_module(&mut world_a);

    let ent_1 = world_a.spawn_auto().unwrap();
    world_a.set(ent_1, SimName("ent_1".to_string())).unwrap();
    world_a.set(ent_1, SimPosition { x: 10.0, y: 20.0 }).unwrap();
    world_a.set(ent_1, SimVelocity { x: 1.0, y: -1.0 }).unwrap();

    let ent_2 = world_a.spawn_auto().unwrap();
    world_a.set(ent_2, SimName("ent_2".to_string())).unwrap();
    world_a.set(ent_2, SimPosition { x: 30.0, y: 40.0 }).unwrap();
    world_a.set(ent_2, SimVelocity { x: -1.0, y: 1.0 }).unwrap();

    // Serialize world_a to JSON and binary ColumnarStoreSnapshot
    let json = store_to_json_world(&world_a);
    elog!("{json}");

    let snapshot_bytes = world_a
        .snapshot()
        .unwrap()
        .into_binary()
        .unwrap();

    // Create second world, import same module, and restore snapshot
    let mut world_b = Store::new();
    import_move_module(&mut world_b);
    let restored_snapshot = ColumnarStoreSnapshot::from_binary(&snapshot_bytes).unwrap();
    world_b.restore_snapshot(&restored_snapshot).unwrap();

    // Run Move system once for both worlds
    run_move_system(&mut world_a);
    elog!();
    run_move_system(&mut world_b);
}

#[test]
fn test_reflection_world_ser_deser() {
    main();
    let lines = take_log();
    assert_eq!(lines.len(), 6);
    let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(parsed["results"].as_array().unwrap().len(), 2);
    assert_eq!(
        &lines[1..],
        &[
            "::ent_1 moved to {x: 11, y: 19}",
            "::ent_2 moved to {x: 29, y: 41}",
            "",
            "::ent_1 moved to {x: 11, y: 19}",
            "::ent_2 moved to {x: 29, y: 41}",
        ]
    );
}
