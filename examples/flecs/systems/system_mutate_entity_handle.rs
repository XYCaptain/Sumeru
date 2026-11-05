//! Port of `examples/flecs/systems/system_mutate_entity_handle.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Timeout {
    pub to_delete: EntityId,
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Tag;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let to_delete = store.entity_named("ToDelete");
    store.set(to_delete, Tag).unwrap();

    let my_entity = store.entity_named("MyEntity");
    store
        .set(
            my_entity,
            Timeout {
                to_delete,
                value: 2.5,
            },
        )
        .unwrap();

    let dt = 1.0_f32;

    loop {
        let mut deleted_targets = Vec::new();

        // System 1: decrement timeout and enqueue target entity deletion when expired
        let mut expired_targets = Vec::new();
        {
            let mut q = store.query_mut::<&mut Timeout>().unwrap();
            for (_entity, timeout) in q.iter() {
                timeout.value -= dt;
                if timeout.value <= 0.0 {
                    expired_targets.push(timeout.to_delete);
                }
            }
        }
        let mut cmd = store.commands();
        for target in expired_targets {
            let name = store.name_of(target);
            elog!("Expire: {} deleted!", name);
            cmd.delete_entity(target);
            deleted_targets.push(name);
        }

        // System 2: print remaining expiry time for target entity (still alive before flush)
        for (_entity, timeout) in store.query::<&Timeout>().iter() {
            elog!(
                "PrintExpire: {} has {:.2} seconds left",
                store.name_of(timeout.to_delete),
                timeout.value
            );
        }

        // End-of-frame sync point: apply deferred deletions and trigger OnRemove observer for Tag
        cmd.apply_discard(&mut store).unwrap();
        for name in deleted_targets {
            elog!("Expired: {} actually deleted", name);
        }

        if !store.is_alive(to_delete) {
            break;
        }

        elog!("Tick...");
    }
}

#[test]
fn test_system_mutate_entity_handle() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "PrintExpire: ToDelete has 1.50 seconds left",
            "Tick...",
            "PrintExpire: ToDelete has 0.50 seconds left",
            "Tick...",
            "Expire: ToDelete deleted!",
            "PrintExpire: ToDelete has -0.50 seconds left",
            "Expired: ToDelete actually deleted",
        ]
    );
}
