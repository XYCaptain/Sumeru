//! Port of `examples/flecs/systems/system_delta_time.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct FrameDeltaTime(pub f32);

fn progress(store: &mut Store, dt_entity: EntityId, dt: f32) {
    store.set(dt_entity, FrameDeltaTime(dt)).unwrap();
    if let Some(frame_dt) = store.get::<FrameDeltaTime>(dt_entity) {
        elog!("delta_time: {}", frame_dt.0);
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    let dt_entity = store.spawn_auto().unwrap();

    // First frame uses 60 FPS default guess (~16.67ms), followed by two ~100ms frames
    progress(&mut store, dt_entity, 1.0 / 60.0);
    progress(&mut store, dt_entity, 0.1);
    progress(&mut store, dt_entity, 0.1);
}

#[test]
fn test_system_delta_time() {
    main();
    let lines = take_log();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with("delta_time: 0.0166666"));
    assert_eq!(lines[1], "delta_time: 0.1");
    assert_eq!(lines[2], "delta_time: 0.1");
}
