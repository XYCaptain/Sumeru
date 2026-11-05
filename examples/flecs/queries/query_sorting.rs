use crate::common::*;
use crate::elog;

// Print values of query sorted by Position.x
fn print_query(store: &Store, q: &Query<&'static Position>) {
    elog!(" -- iterate query");
    let mut items: Vec<Position> = Vec::new();
    q.for_each(store, |(_e, pos)| {
        items.push(*pos);
    })
    .unwrap();
    items.sort_by(|p1, p2| p1.x.total_cmp(&p2.x));
    for pos in items {
        elog!("{{{},{}}}", pos.x, pos.y);
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entities, set Position indifferent order
    let e = store.spawn_empty();
    store.set(e, Position { x: 1.0, y: 0.0 });

    let e2 = store.spawn_empty();
    store.set(e2, Position { x: 6.0, y: 0.0 });

    let e3 = store.spawn_empty();
    store.set(e3, Position { x: 2.0, y: 0.0 });

    let e4 = store.spawn_empty();
    store.set(e4, Position { x: 5.0, y: 0.0 });

    let e5 = store.spawn_empty();
    store.set(e5, Position { x: 4.0, y: 0.0 });

    // Build a query for Position
    let q = QueryBuilder::<&Position>::new().build(&store).unwrap();

    // First iteration: data is sorted
    print_query(&store, &q);

    // Change the value of one entity, invalidating the order
    store.set(e, Position { x: 7.0, y: 0.0 });

    // Second iteration: data is resorted
    print_query(&store, &q);

    // Create new entity to show that data is also sorted for system
    let e6 = store.spawn_empty();
    store.set(e6, Position { x: 3.0, y: 0.0 });

    // Run system, output will be sorted
    elog!("\n -- run system");
    let mut sys_items: Vec<Position> = store
        .query::<&Position>()
        .iter()
        .map(|(_e, pos)| *pos)
        .collect();
    sys_items.sort_by(|p1, p2| p1.x.total_cmp(&p2.x));
    for pos in sys_items {
        elog!("{{{},{}}}", pos.x, pos.y);
    }
}

#[test]
fn test_query_sorting() {
    main();
    assert_eq!(
        take_log(),
        vec![
            " -- iterate query",
            "{1,0}",
            "{2,0}",
            "{4,0}",
            "{5,0}",
            "{6,0}",
            " -- iterate query",
            "{2,0}",
            "{4,0}",
            "{5,0}",
            "{6,0}",
            "{7,0}",
            "\n -- run system",
            "{2,0}",
            "{3,0}",
            "{4,0}",
            "{5,0}",
            "{6,0}",
            "{7,0}",
        ]
    );
}
