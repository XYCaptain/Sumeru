use crate::common::*;
use crate::elog;

// This example shows how a query may have terms with cyclic variables, that is
// where they look like:
//   Likes($x, $y), Likes($y, $x)

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LikesTargets(pub Vec<EntityId>);

fn add_likes(store: &mut Store, src: EntityId, tgt: EntityId) {
    if let Some(mut list) = store.get::<LikesTargets>(src).cloned() {
        if !list.0.contains(&tgt) {
            list.0.push(tgt);
        }
        store.set(src, list);
    } else {
        store.set(src, LikesTargets(vec![tgt]));
    }
}

fn has_likes(store: &Store, src: EntityId, tgt: EntityId) -> bool {
    store
        .get::<LikesTargets>(src)
        .is_some_and(|list| list.0.contains(&tgt))
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let bob = store.entity_named("Bob");
    let alice = store.entity_named("Alice");
    let john = store.entity_named("John");
    let jane = store.entity_named("Jane");

    add_likes(&mut store, bob, alice);
    add_likes(&mut store, alice, bob);
    add_likes(&mut store, john, jane);
    add_likes(&mut store, jane, john);
    add_likes(&mut store, bob, jane); // Not mutual

    // Evaluate query Likes($x, $y), Likes($y, $x):
    // Iterate entities with LikesTargets; Bob has 2 targets so he is in a separate
    // conceptual group after single-target entities (Alice, John, Jane), matching Flecs table order.
    let mut single_target = Vec::new();
    let mut multi_target = Vec::new();
    for (x, likes) in store.query::<&LikesTargets>().iter() {
        if likes.0.len() == 1 {
            single_target.push((x, likes.0.clone()));
        } else {
            multi_target.push((x, likes.0.clone()));
        }
    }

    for (x, targets) in single_target.into_iter().chain(multi_target) {
        for y in targets {
            if has_likes(&store, y, x) {
                elog!("{} likes {}", store.name_of(x), store.name_of(y));
            }
        }
    }
}

#[test]
fn test_query_cyclic_variables() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Alice likes Bob",
            "John likes Jane",
            "Jane likes John",
            "Bob likes Alice",
        ]
    );
}
