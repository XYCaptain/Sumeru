//! Port of `examples/flecs/systems/system_custom_phases_no_builtin.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PhaseTag;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct DependsOn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct SystemKind;

fn phase_depth(store: &Store, phase: EntityId) -> usize {
    let mut depth = 0;
    let mut cur = store.pair_target::<DependsOn>(phase);
    while let Some(parent) = cur {
        depth += 1;
        cur = store.pair_target::<DependsOn>(parent);
    }
    depth
}

fn sys(store: &Store, system_entity: EntityId) {
    elog!("system {}", store.name_of(system_entity));
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Three custom phases with no builtin phase
    let update = store.spawn_auto().unwrap();
    store.set(update, PhaseTag).unwrap();

    let physics = store.spawn_auto().unwrap();
    store.set(physics, PhaseTag).unwrap();
    store.set_pair(physics, DependsOn, update).unwrap();

    let collisions = store.spawn_auto().unwrap();
    store.set(collisions, PhaseTag).unwrap();
    store.set_pair(collisions, DependsOn, physics).unwrap();

    // Create 3 dummy systems
    let collision_sys = store.entity_named("CollisionSystem");
    store.set_pair(collision_sys, SystemKind, collisions).unwrap();

    let physics_sys = store.entity_named("PhysicsSystem");
    store.set_pair(physics_sys, SystemKind, physics).unwrap();

    let game_sys = store.entity_named("GameSystem");
    store.set_pair(game_sys, SystemKind, update).unwrap();

    // Run pipeline ordered by custom phase dependency chain
    let mut scheduled = Vec::new();
    for (sys_entity, kind_pair) in store.query::<&Pair<SystemKind, EntityId>>().iter() {
        let phase = kind_pair.target;
        if store.has::<PhaseTag>(phase) {
            scheduled.push((phase_depth(&store, phase), sys_entity));
        }
    }
    scheduled.sort_by_key(|(depth, _)| *depth);

    for (_, sys_entity) in scheduled {
        sys(&store, sys_entity);
    }
}

#[test]
fn test_system_custom_phases_no_builtin() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "system GameSystem",
            "system PhysicsSystem",
            "system CollisionSystem",
        ]
    );
}
