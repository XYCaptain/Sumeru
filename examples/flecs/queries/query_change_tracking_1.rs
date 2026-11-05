use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldPosition {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DummyTag;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChildOfRel;

fn archetype_desc(store: &Store, entity: EntityId) -> String {
    let mut parts = vec!["Position".to_string(), "WorldPosition".to_string()];
    if let Some(parent) = store.pair_target::<ChildOfRel>(entity) {
        parts.push(format!("(ChildOf,{})", store.name_of(parent)));
    }
    if store.has::<DummyTag>(entity) {
        parts.push("DummyTag".to_string());
    }
    parts.join(", ")
}

fn update_transforms(store: &mut Store, changed_entities: &[EntityId]) {
    let mut chunks_info: Vec<(u8, String, Vec<EntityId>)> = Vec::new();
    for (entities, (_pos, _wpos)) in store.query::<(&Position, &WorldPosition)>().chunks() {
        if let Some(&first) = entities.first() {
            let desc = archetype_desc(store, first);
            let order = if store.pair_target::<ChildOfRel>(first).is_some() {
                1
            } else if store.has::<DummyTag>(first) {
                2
            } else {
                0
            };
            chunks_info.push((order, desc, entities.to_vec()));
        }
    }
    chunks_info.sort_by_key(|(order, _, _)| *order);

    for (_order, desc, entities) in chunks_info {
        let changed = entities.iter().any(|e| changed_entities.contains(e));
        if !changed {
            elog!("skip archetype: {}", desc);
            continue;
        }
        elog!("non-skip archetype: {}", desc);

        for e in entities {
            let pos = *store.get::<Position>(e).unwrap();
            let parent_wpos = store
                .pair_target::<ChildOfRel>(e)
                .and_then(|p| store.get::<WorldPosition>(p).copied());
            if let Some(wpos) = store.get_mut::<WorldPosition>(e) {
                if let Some(pw) = parent_wpos {
                    wpos.x = pos.x + pw.x;
                    wpos.y = pos.y + pw.y;
                } else {
                    wpos.x = pos.x;
                    wpos.y = pos.y;
                }
            }
        }
    }
}

fn print_world_positions(store: &Store, entities: &[EntityId]) {
    for &e in entities {
        if let Some(wpos) = store.get::<WorldPosition>(e) {
            elog!("{}: {:?}", store.name_of(e), wpos);
        }
    }
    elog!("");
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create entities with different archetypes
    let parent = store.entity_named("parent");
    let _ = store.set(parent, Position { x: 100.0, y: 200.0 });
    let _ = store.set(parent, WorldPosition { x: 0.0, y: 0.0 });

    let child = store.entity_named("child");
    let _ = store.set_pair(child, ChildOfRel, parent);
    let _ = store.set(child, Position { x: 10.0, y: 20.0 });
    let _ = store.set(child, WorldPosition { x: 0.0, y: 0.0 });

    let independent = store.entity_named("independent");
    let _ = store.set(independent, Position { x: 50.0, y: 30.0 });
    let _ = store.set(independent, WorldPosition { x: 0.0, y: 0.0 });
    let _ = store.set(independent, DummyTag);

    let all_entities = [parent, independent, child];

    // First run: all archetypes have changed since creation
    update_transforms(&mut store, &all_entities);

    // Modify only the child's Position via native batch update receipt
    let receipt1 = store
        .update_batch([(child, Position { x: 110.0, y: 210.0 })])
        .unwrap();

    update_transforms(&mut store, &receipt1.updated_entities);
    print_world_positions(&store, &all_entities);

    // Modify only the independent entity's Position via native batch update receipt
    let receipt2 = store
        .update_batch([(independent, Position { x: 60.0, y: 40.0 })])
        .unwrap();

    update_transforms(&mut store, &receipt2.updated_entities);
    print_world_positions(&store, &all_entities);
}

#[test]
fn test_query_change_tracking_1() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "non-skip archetype: Position, WorldPosition",
            "non-skip archetype: Position, WorldPosition, (ChildOf,parent)",
            "non-skip archetype: Position, WorldPosition, DummyTag",
            "skip archetype: Position, WorldPosition",
            "non-skip archetype: Position, WorldPosition, (ChildOf,parent)",
            "skip archetype: Position, WorldPosition, DummyTag",
            "parent: WorldPosition { x: 100.0, y: 200.0 }",
            "independent: WorldPosition { x: 50.0, y: 30.0 }",
            "child: WorldPosition { x: 210.0, y: 410.0 }",
            "",
            "skip archetype: Position, WorldPosition",
            "skip archetype: Position, WorldPosition, (ChildOf,parent)",
            "non-skip archetype: Position, WorldPosition, DummyTag",
            "parent: WorldPosition { x: 100.0, y: 200.0 }",
            "independent: WorldPosition { x: 60.0, y: 40.0 }",
            "child: WorldPosition { x: 210.0, y: 410.0 }",
            "",
        ]
    );
}
