use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a few test entities for a Position, Velocity query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 });
    store.set(e1, Velocity { x: 1.0, y: 2.0 });

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 10.0, y: 20.0 });
    store.set(e2, Velocity { x: 3.0, y: 4.0 });

    // This entity will not match as it does not have Position, Velocity
    let e3 = store.entity_named("e3");
    store.set(e3, Position { x: 10.0, y: 20.0 });

    // Ad hoc queries are a bit slower to iterate than cached `Query` handles, since
    // the tables can't be pre-cached, but they are fast to create and don't require
    // any ahead-of-time registration.
    for (_entity, (name, pos, vel)) in store
        .query_mut::<(&EntityName, &mut Position, &Velocity)>()
        .unwrap()
        .iter()
    {
        pos.x += vel.x;
        pos.y += vel.y;
        elog!("Entity {}: {:?}", name.0, pos);
    }
}

#[test]
fn test_query_world_query() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Entity e1: Position { x: 11.0, y: 22.0 }",
            "Entity e2: Position { x: 13.0, y: 24.0 }",
        ]
    );
}
