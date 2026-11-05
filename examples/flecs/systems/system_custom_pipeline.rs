//! Port of `examples/flecs/systems/system_custom_pipeline.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Physics;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMessage(pub String);

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a custom pipeline query that only matches systems with `Physics`
    let pipeline = QueryBuilder::<&SystemMessage>::new()
        .with::<Physics>()
        .build(&store)
        .unwrap();

    // Create system with Physics tag
    let sys_with_physics = store.spawn_auto().unwrap();
    store
        .set(
            sys_with_physics,
            SystemMessage("System with Physics ran!".to_string()),
        )
        .unwrap();
    store.set(sys_with_physics, Physics).unwrap();

    // Create system without Physics tag
    let sys_without_physics = store.spawn_auto().unwrap();
    store
        .set(
            sys_without_physics,
            SystemMessage("System without Physics ran!".to_string()),
        )
        .unwrap();

    // Run custom pipeline: only systems matched by `pipeline` execute
    pipeline
        .for_each(&store, |(_e, msg)| {
            elog!("{}", msg.0);
        })
        .unwrap();
}

#[test]
fn test_system_custom_pipeline() {
    main();
    assert_eq!(take_log(), vec!["System with Physics ran!"]);
}
