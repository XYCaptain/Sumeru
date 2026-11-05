use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionI32 {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VelocityI32 {
    pub x: i32,
    pub y: i32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Set singleton Position
    store.set_singleton(PositionI32 { x: 1, y: 2 }).unwrap();

    // Spawn entity with Velocity
    let e = store.spawn_empty();
    store.set(e, VelocityI32 { x: 590, y: 20 });

    // Build reusable query and pass it into a system closure
    let query = QueryBuilder::<&'static VelocityI32>::new().build(&store).unwrap();

    let run_system = |store: &Store, q: &Query<&'static VelocityI32>| {
        if let Some(pos) = store.singleton::<PositionI32>() {
            q.for_each(store, |(_entity, vel)| {
                elog!(
                    "Position {{ x: {}, y: {} }}, Velocity {{ x: {}, y: {} }}",
                    pos.x,
                    pos.y,
                    vel.x,
                    vel.y
                );
            })
            .unwrap();
        }
    };

    run_system(&store, &query);
}

#[test]
fn test_query_pass_query_to_system() {
    main();
    assert_eq!(
        take_log(),
        vec!["Position { x: 1, y: 2 }, Velocity { x: 590, y: 20 }"]
    );
}
