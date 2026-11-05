use crate::common::*;
use crate::elog;

// Transitive relationships make it possible to tell the ECS that if an entity
// has a relationship (R, X) and X has relationship (R, Y), the entity should be
// treated as if it also has (R, Y).

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocatedIn;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Planet;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Continent;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Country;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct State;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct City;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person;

fn find_transitive_location_with<T: Component>(store: &Store, start: EntityId) -> Option<EntityId> {
    let mut current = store.pair_target::<LocatedIn>(start);
    while let Some(loc) = current {
        if store.has::<T>(loc) {
            return Some(loc);
        }
        current = store.pair_target::<LocatedIn>(loc);
    }
    None
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Populate the store with locations
    let earth = store.entity_named("Earth");
    store.set(earth, Planet);

    // Continents
    let north_america = store.entity_named("NorthAmerica");
    store.set(north_america, Continent);
    store.set_pair(north_america, LocatedIn, earth).unwrap();

    let europe = store.entity_named("Europe");
    store.set(europe, Continent);
    store.set_pair(europe, LocatedIn, earth).unwrap();

    // Countries
    let united_states = store.entity_named("UnitedStates");
    store.set(united_states, Country);
    store.set_pair(united_states, LocatedIn, north_america).unwrap();

    let netherlands = store.entity_named("Netherlands");
    store.set(netherlands, Country);
    store.set_pair(netherlands, LocatedIn, europe).unwrap();

    // States
    let california = store.entity_named("California");
    store.set(california, State);
    store.set_pair(california, LocatedIn, united_states).unwrap();

    let washington = store.entity_named("Washington");
    store.set(washington, State);
    store.set_pair(washington, LocatedIn, united_states).unwrap();

    let noord_holland = store.entity_named("NoordHolland");
    store.set(noord_holland, State);
    store.set_pair(noord_holland, LocatedIn, netherlands).unwrap();

    // Cities
    let san_francisco = store.entity_named("SanFrancisco");
    store.set(san_francisco, City);
    store.set_pair(san_francisco, LocatedIn, california).unwrap();

    let seattle = store.entity_named("Seattle");
    store.set(seattle, City);
    store.set_pair(seattle, LocatedIn, washington).unwrap();

    let amsterdam = store.entity_named("Amsterdam");
    store.set(amsterdam, City);
    store.set_pair(amsterdam, LocatedIn, noord_holland).unwrap();

    // Inhabitants
    let bob = store.entity_named("Bob");
    store.set(bob, Person);
    store.set_pair(bob, LocatedIn, san_francisco).unwrap();

    let alice = store.entity_named("Alice");
    store.set(alice, Person);
    store.set_pair(alice, LocatedIn, seattle).unwrap();

    let job = store.entity_named("Job");
    store.set(job, Person);
    store.set_pair(job, LocatedIn, amsterdam).unwrap();

    // Query: Person, (LocatedIn, $Location), Country($Location)
    // Because LocatedIn is transitive, this traverses upward to find the Country each person lives in.
    for (person, _) in store.query::<&Person>().iter() {
        if let Some(country) = find_transitive_location_with::<Country>(&store, person) {
            elog!("{} lives in {}", store.name_of(person), store.name_of(country));
        }
    }
}

#[test]
fn test_query_transitive_queries() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Bob lives in UnitedStates",
            "Alice lives in UnitedStates",
            "Job lives in Netherlands",
        ]
    );
}
