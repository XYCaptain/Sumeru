//! Port of `examples/flecs/systems/system_target_fps.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct TargetFps(pub f32);

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Store target FPS configuration on a world config entity
    let cfg = store.spawn_auto().unwrap();
    store.set(cfg, TargetFps(1.0)).unwrap();

    // Run 3 frames at target FPS = 1.0 (delta_time = 1.0 / target_fps)
    for _ in 0..3 {
        let fps = store.get::<TargetFps>(cfg).unwrap().0;
        let dt = 1.0 / fps;
        elog!("Delta time: {}", dt);
    }
}

#[test]
fn test_system_target_fps() {
    main();
    assert_eq!(
        take_log(),
        vec!["Delta time: 1", "Delta time: 1", "Delta time: 1"]
    );
}
