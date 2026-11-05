use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EatsList(pub Vec<EntityId>);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Healthy;

fn add_eats(store: &mut Store, eater: EntityId, food: EntityId) {
    if let Some(mut list) = store.get::<EatsList>(eater).cloned() {
        if !list.0.contains(&food) {
            list.0.push(food);
        }
        store.set(eater, list);
    } else {
        store.set(eater, EatsList(vec![food]));
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let healthy = store.entity_named("Healthy");
    let _ = healthy;

    let apples = store.entity_named("Apples");
    store.set(apples, Healthy);

    let salad = store.entity_named("Salad");
    store.set(salad, Healthy);

    let burgers = store.entity_named("Burgers");
    let pizza = store.entity_named("Pizza");
    let chocolate = store.entity_named("Chocolate");

    let bob = store.entity_named("Bob");
    add_eats(&mut store, bob, apples);
    add_eats(&mut store, bob, burgers);
    add_eats(&mut store, bob, pizza);

    let alice = store.entity_named("Alice");
    add_eats(&mut store, alice, salad);
    add_eats(&mut store, alice, chocolate);
    add_eats(&mut store, alice, apples);

    // Here we're creating a rule that in the query DSL would look like this:
    //   Eats($this, $Food), Healthy($Food)
    // Iterate by healthy food target (Apples, then Salad) to match Flecs variable binding order.
    let healthy_foods: Vec<EntityId> = store
        .query::<&Healthy>()
        .iter()
        .map(|(food, _)| food)
        .collect();

    for food in healthy_foods {
        for (eater, eats) in store.query::<&EatsList>().iter() {
            if eats.0.contains(&food) {
                elog!("{} eats {}", store.name_of(eater), store.name_of(food));
            }
        }
    }
}

#[test]
fn test_query_variables() {
    main();
    assert_eq!(
        take_log(),
        vec!["Bob eats Apples", "Alice eats Apples", "Alice eats Salad"]
    );
}
