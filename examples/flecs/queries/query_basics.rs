use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a query for Position, Velocity.
    let query = QueryBuilder::<(&mut Position, &Velocity)>::new()
        .build(&store)
        .unwrap();

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

    // 1. each_entity equivalent: iterate with entity id and name
    for (_e, (name, pos, vel)) in store
        .query_mut::<(&EntityName, &mut Position, &Velocity)>()
        .unwrap()
        .iter()
    {
        pos.x += vel.x;
        pos.y += vel.y;
        elog!("{}: [{:?}]", name.0, pos);
    }

    // 2. each equivalent: iterate components directly via reusable Query handle
    query
        .for_each_mut(&mut store, |(_e, (pos, vel))| {
            pos.x += vel.x;
            pos.y += vel.y;
            elog!("[{pos:?}]");
        })
        .unwrap();

    // 3. run equivalent: low-level chunk/table iteration
    for (_entities, (p, v)) in store
        .query_mut::<(&mut Position, &Velocity)>()
        .unwrap()
        .chunks()
    {
        for i in 0..p.len() {
            p[i].x += v[i].x;
            p[i].y += v[i].y;
            elog!("[{:?}]", p[i]);
        }
    }
}

#[test]
fn test_query_basics() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "e1: [Position { x: 11.0, y: 22.0 }]",
            "e2: [Position { x: 13.0, y: 24.0 }]",
            "[Position { x: 12.0, y: 24.0 }]",
            "[Position { x: 16.0, y: 28.0 }]",
            "[Position { x: 13.0, y: 26.0 }]",
            "[Position { x: 19.0, y: 32.0 }]",
        ]
    );
}
