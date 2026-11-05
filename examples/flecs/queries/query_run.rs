use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Mass {
    pub value: f32,
}

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

    // This entity is in a different table because it also has Mass
    let e3 = store.entity_named("e3");
    store.set(e3, Position { x: 10.0, y: 20.0 });
    store.set(e3, Velocity { x: 4.0, y: 5.0 });
    store.set(e3, Mass { value: 50.0 });

    // The run() equivalent provides direct chunk-level control over table iteration
    elog!("MoveEntities: begin");

    // Collect table metadata before mutating chunks
    let has_mass_flags: Vec<bool> = store
        .query::<(&Position, &Velocity)>()
        .chunks()
        .map(|(entities, _)| entities.first().is_some_and(|&e| store.has::<Mass>(e)))
        .collect();

    for ((entities, (names, p, v)), has_mass) in store
        .query_mut::<(&EntityName, &mut Position, &Velocity)>()
        .unwrap()
        .chunks()
        .zip(has_mass_flags)
    {
        let table_str = if has_mass {
            "Position, Velocity, Mass, (Identifier,Name)"
        } else {
            "Position, Velocity, (Identifier,Name)"
        };
        elog!(" - table [{}]", table_str);
        elog!(" - number of entities: {}", entities.len());
        elog!(" - term 1: ");
        elog!("   - component: Position");
        elog!("   - type size: {}", std::mem::size_of::<Position>());
        elog!(" - term 2: ");
        elog!("   - component: Velocity");
        elog!("   - type size: {}", std::mem::size_of::<Velocity>());
        elog!("");

        for i in 0..entities.len() {
            p[i].x += v[i].x;
            p[i].y += v[i].y;
            elog!(" - {}: {:?}", names[i].0, p[i]);
        }

        elog!("");
    }

    elog!("MoveEntities: end");
}

#[test]
fn test_query_run() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "MoveEntities: begin",
            " - table [Position, Velocity, (Identifier,Name)]",
            " - number of entities: 2",
            " - term 1: ",
            "   - component: Position",
            "   - type size: 8",
            " - term 2: ",
            "   - component: Velocity",
            "   - type size: 8",
            "",
            " - e1: Position { x: 11.0, y: 22.0 }",
            " - e2: Position { x: 13.0, y: 24.0 }",
            "",
            " - table [Position, Velocity, Mass, (Identifier,Name)]",
            " - number of entities: 1",
            " - term 1: ",
            "   - component: Position",
            "   - type size: 8",
            " - term 2: ",
            "   - component: Velocity",
            "   - type size: 8",
            "",
            " - e3: Position { x: 14.0, y: 25.0 }",
            "",
            "MoveEntities: end",
        ]
    );
}
