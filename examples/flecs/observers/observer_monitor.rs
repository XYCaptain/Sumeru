//! Port of `examples/flecs/observers/observer_monitor.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

fn matches_monitor(store: &Store, entity: EntityId) -> bool {
    store.has::<Position>(entity) && store.has::<Velocity>(entity)
}

fn check_monitor_transition(
    store: &Store,
    entity: EntityId,
    was_matching: bool,
    changed_component: &str,
) {
    let now_matching = matches_monitor(store, entity);
    if !was_matching && now_matching {
        elog!(" - Enter: {}: {}", changed_component, store.name_of(entity));
    } else if was_matching && !now_matching {
        elog!(" - Leave: {}: {}", changed_component, store.name_of(entity));
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entity
    let entity = store.entity_named("e");

    // Setting Position alone does not yet match (&Position, &Velocity)
    let before = matches_monitor(&store, entity);
    store.set(entity, Position { x: 10.0, y: 20.0 }).unwrap();
    check_monitor_transition(&store, entity, before, "Position");

    // Setting Velocity causes entity to enter the (&Position, &Velocity) monitor
    let before = matches_monitor(&store, entity);
    store.set(entity, Velocity { x: 1.0, y: 2.0 }).unwrap();
    check_monitor_transition(&store, entity, before, "Velocity");

    // Removing Position causes entity to leave the (&Position, &Velocity) monitor
    let before = matches_monitor(&store, entity);
    store.remove::<Position>(entity).unwrap();
    check_monitor_transition(&store, entity, before, "Position");
}

#[test]
fn test_observer_monitor() {
    main();
    assert_eq!(
        take_log(),
        vec![" - Enter: Velocity: e", " - Leave: Position: e"]
    );
}
