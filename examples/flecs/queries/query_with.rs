use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Npc;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a query for Position, Npc. By adding the Npc component using the
    // with method, the component is not a part of the query type, and as a
    // result does not become part of the function signatures of each callbacks.
    // This is useful for things like tags, which because they don't have a
    // value are less useful to pass to the each callback as argument.
    let query = QueryBuilder::<&Position>::new()
        .with::<Npc>()
        .build(&store)
        .unwrap();

    // Create a few test entities for the Position, Npc query
    let e1 = store.entity_named("e1");
    store.set(e1, Position { x: 10.0, y: 20.0 });
    store.set(e1, Npc);

    let e2 = store.entity_named("e2");
    store.set(e2, Position { x: 10.0, y: 20.0 });
    store.set(e2, Npc);

    // This entity will not match as it does not have Position, Npc
    let e3 = store.entity_named("e3");
    store.set(e3, Position { x: 10.0, y: 20.0 });

    // Note how the Npc tag is not part of the closure signature
    query
        .for_each(&store, |(entity, pos)| {
            elog!("Entity {}: {:?}", store.name_of(entity), pos);
        })
        .unwrap();
}

#[test]
fn test_query_with() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Entity e1: Position { x: 10.0, y: 20.0 }",
            "Entity e2: Position { x: 10.0, y: 20.0 }",
        ]
    );
}
