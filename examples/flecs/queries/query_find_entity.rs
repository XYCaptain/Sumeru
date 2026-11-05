use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 });

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 20.0, y: 30.0 });

    // Find the entity whose Position.x == 20.0
    let entity = store.find_entity_by::<Position>(|pos| (pos.x - 20.0).abs() < f32::EPSILON);

    if let Some(entity) = entity {
        elog!("Entity found: {:?}", store.path_of(entity));
    } else {
        elog!("Entity not found");
    }
}

#[test]
fn test_query_find_entity() {
    main();
    assert_eq!(take_log(), vec!["Entity found: \"::e2\""]);
}
