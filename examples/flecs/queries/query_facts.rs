use crate::common::*;
use crate::elog;

// This example shows how to use queries for testing facts. A fact is a query that
// has no entity result, where each term has a fixed source and target:
//   Likes($x, $y), Likes($y, $x) evaluated for given ($x, $y).

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

fn are_friends(store: &Store, x: EntityId, y: EntityId) -> bool {
    let x_likes_y = store
        .get::<LikesTargets>(x)
        .is_some_and(|list| list.0.contains(&y));
    let y_likes_x = store
        .get::<LikesTargets>(y)
        .is_some_and(|list| list.0.contains(&x));
    x_likes_y && y_likes_x
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let bob = store.entity_named("Bob");
    let alice = store.entity_named("Alice");
    let jane = store.entity_named("Jane");
    let john = store.entity_named("John");

    add_likes(&mut store, bob, alice);
    add_likes(&mut store, alice, bob);
    add_likes(&mut store, jane, john);
    add_likes(&mut store, john, jane);
    add_likes(&mut store, bob, john); // nope!

    elog!(
        "Are Bob and Alice friends? {}",
        if are_friends(&store, bob, alice) {
            "Yes"
        } else {
            "No"
        }
    );
    elog!(
        "Are Bob and John friends? {}",
        if are_friends(&store, bob, john) {
            "Yes"
        } else {
            "No"
        }
    );
    elog!(
        "Are Jane and John friends? {}",
        if are_friends(&store, jane, john) {
            "Yes"
        } else {
            "No"
        }
    );
    elog!(
        "Are John and Jane friends? {}",
        if are_friends(&store, john, jane) {
            "Yes"
        } else {
            "No"
        }
    );
}

#[test]
fn test_query_facts() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Are Bob and Alice friends? Yes",
            "Are Bob and John friends? No",
            "Are Jane and John friends? Yes",
            "Are John and Jane friends? Yes",
        ]
    );
}
