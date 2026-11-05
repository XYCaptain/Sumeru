use crate::common::*;
use crate::elog;

// Queries can have wildcard terms that can match multiple instances of a
// relationship or relationship target.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EatsAmount {
    pub amount: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Apples;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pears;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let bob = store.entity_named("Bob");
    store.set(bob, Pair::new(EatsAmount { amount: 10 }, Apples));
    store.set(bob, Pair::new(EatsAmount { amount: 5 }, Pears));

    let alice = store.entity_named("Alice");
    store.set(alice, Pair::new(EatsAmount { amount: 4 }, Apples));

    // Match (EatsAmount, *) wildcard across all food targets:
    // Alice has only (EatsAmount, Apples), while Bob has both (EatsAmount, Apples) and (EatsAmount, Pears)
    // so Alice's single-pair table is visited before Bob's two-pair table.
    let mut single_food = Vec::new();
    let mut multi_food = Vec::new();

    for (e, apples) in store.query::<&Pair<EatsAmount, Apples>>().iter() {
        if let Some(pears) = store.get::<Pair<EatsAmount, Pears>>(e) {
            multi_food.push((e, "Apples", apples.relation.amount));
            multi_food.push((e, "Pears", pears.relation.amount));
        } else {
            single_food.push((e, "Apples", apples.relation.amount));
        }
    }

    for (e, food_name, amount) in single_food.into_iter().chain(multi_food) {
        elog!("{} eats {} {}", store.name_of(e), amount, food_name);
    }
}

#[test]
fn test_query_wildcard() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Alice eats 4 Apples",
            "Bob eats 10 Apples",
            "Bob eats 5 Pears",
        ]
    );
}
