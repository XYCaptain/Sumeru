use crate::common::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationPairs(pub Vec<(EntityId, EntityId)>);

fn add_rel(store: &mut Store, entity: EntityId, rel: EntityId, target: EntityId) {
    let mut pairs = store
        .get::<RelationPairs>(entity)
        .cloned()
        .unwrap_or_default();
    if !pairs.0.contains(&(rel, target)) {
        pairs.0.push((rel, target));
    }
    store.set(entity, pairs).unwrap();
}

fn has_rel(store: &Store, entity: EntityId, rel: EntityId, target: EntityId) -> bool {
    store
        .get::<RelationPairs>(entity)
        .map(|p| p.0.contains(&(rel, target)))
        .unwrap_or(false)
}

fn has_rel_wildcard_target(store: &Store, entity: EntityId, rel: EntityId) -> bool {
    store
        .get::<RelationPairs>(entity)
        .map(|p| p.0.iter().any(|(r, _)| *r == rel))
        .unwrap_or(false)
}

fn targets_for_rel(store: &Store, entity: EntityId, rel: EntityId) -> Vec<EntityId> {
    store
        .get::<RelationPairs>(entity)
        .map(|p| {
            p.0.iter()
                .filter(|(r, _)| *r == rel)
                .map(|(_, t)| *t)
                .collect()
        })
        .unwrap_or_default()
}

fn rels_for_target(store: &Store, entity: EntityId, target: EntityId) -> Vec<EntityId> {
    store
        .get::<RelationPairs>(entity)
        .map(|p| {
            p.0.iter()
                .filter(|(_, t)| *t == target)
                .map(|(r, _)| *r)
                .collect()
        })
        .unwrap_or_default()
}

fn format_type(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if store.has::<EntityName>(entity) {
        parts.push("(Identifier,Name)".to_string());
    }
    if let Some(pairs) = store.get::<RelationPairs>(entity) {
        for (rel, target) in &pairs.0 {
            let rel_name = store.name_of(*rel);
            let target_name = store.name_of(*target);
            parts.push(format!("({rel_name},{target_name})"));
        }
    }
    format!("[{}]", parts.join(", "))
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let eats = store.entity_named("Eats");
    let grows = store.entity_named("Grows");
    let apples = store.entity_named("Apples");
    let pears = store.entity_named("Pears");

    let bob = store.entity_named("Bob");
    add_rel(&mut store, bob, eats, apples);
    add_rel(&mut store, bob, eats, pears);
    add_rel(&mut store, bob, grows, pears);

    elog!("Bob eats apples? {}", has_rel(&store, bob, eats, apples));
    elog!(
        "Bob grows food? {}, {}",
        has_rel_wildcard_target(&store, bob, grows),
        has_rel_wildcard_target(&store, bob, grows)
    );
    elog!();

    elog!("Bob's type: {}", format_type(&store, bob));
    elog!();

    for target in targets_for_rel(&store, bob, eats) {
        elog!("Bob eats {}", store.name_of(target));
    }
    elog!();

    for rel in rels_for_target(&store, bob, pears) {
        elog!("Bob {} pears", store.name_of(rel));
    }
    elog!();

    let eat_targets = targets_for_rel(&store, bob, eats);
    if let Some(first) = eat_targets.first() {
        elog!("Bob eats {}", store.name_of(*first));
    }
    if let Some(second) = eat_targets.get(1) {
        elog!("Bob also eats {}", store.name_of(*second));
    }
}

#[test]
fn test_relationships_basics() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "Bob eats apples? true\n",
            "Bob grows food? true, true\n",
            "\n",
            "Bob's type: [(Identifier,Name), (Eats,Apples), (Eats,Pears), (Grows,Pears)]\n",
            "\n",
            "Bob eats Apples\n",
            "Bob eats Pears\n",
            "\n",
            "Bob Eats pears\n",
            "Bob Grows pears\n",
            "\n",
            "Bob eats Apples\n",
            "Bob also eats Pears"
        )
    );
}
