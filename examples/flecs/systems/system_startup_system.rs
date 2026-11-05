//! Port of `examples/flecs/systems/system_startup_system.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    OnStart,
    OnUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemPhase(pub Phase);

fn progress(store: &Store, frame_idx: usize) {
    if frame_idx == 0 {
        for (e, phase) in store.query::<&SystemPhase>().iter() {
            if phase.0 == Phase::OnStart {
                elog!("{}", store.name_of(e));
            }
        }
    }
    for (e, phase) in store.query::<&SystemPhase>().iter() {
        if phase.0 == Phase::OnUpdate {
            elog!("{}", store.name_of(e));
        }
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Startup system registered in OnStart phase
    let startup = store.entity_named("Startup");
    store.set(startup, SystemPhase(Phase::OnStart)).unwrap();

    // Regular system registered in OnUpdate phase
    let update = store.entity_named("Update");
    store.set(update, SystemPhase(Phase::OnUpdate)).unwrap();

    // First frame runs both Startup and Update systems
    progress(&store, 0);

    // Second frame runs only the Update system
    progress(&store, 1);
}

#[test]
fn test_system_startup_system() {
    main();
    assert_eq!(take_log(), vec!["Startup", "Update", "Update"]);
}
