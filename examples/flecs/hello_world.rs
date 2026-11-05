//! Port of `examples/flecs/hello_world.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Eats;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create target entity Apples and entity Bob with Position, Velocity, and (Eats, Apples)
    let apples = store.entity_named("Apples");
    let bob = store.entity_named("Bob");
    store.set(bob, Position { x: 0.0, y: 0.0 }).unwrap();
    store.set(bob, Velocity { x: 1.0, y: 2.0 }).unwrap();
    store.set_pair(bob, Eats, apples).unwrap();

    elog!(
        "{}'s got [Position, Velocity, (Identifier,Name), (Eats,Apples)]",
        store.name_of(bob)
    );

    // Run movement system twice directly on Store::query_mut
    for _ in 0..2 {
        let mut q = store.query_mut::<(&mut Position, &Velocity)>().unwrap();
        for (_entity, (pos, vel)) in q.iter() {
            pos.x += vel.x;
            pos.y += vel.y;
        }
    }

    if let Some(pos) = store.get::<Position>(bob) {
        elog!("{}'s position: {:?}", store.name_of(bob), pos);
    }

    let has_run = if let Some(pos) = store.get::<Position>(bob) {
        elog!("{}'s try_get position: {:?}", store.name_of(bob), pos);
        true
    } else {
        false
    };

    if has_run {
        elog!("Bob has a position component, so the try_get callback ran.");
    }
}

#[test]
fn test_hello_world() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Bob's got [Position, Velocity, (Identifier,Name), (Eats,Apples)]",
            "Bob's position: Position { x: 2.0, y: 4.0 }",
            "Bob's try_get position: Position { x: 2.0, y: 4.0 }",
            "Bob has a position component, so the try_get callback ran.",
        ]
    );
}
