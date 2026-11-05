//! Port of `examples/flecs/systems/system_mutate_entity.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Timeout {
    pub value: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let e = store.entity_named("MyEntity");
    store.set(e, Timeout { value: 2.5 }).unwrap();

    let dt = 1.0_f32;

    loop {
        let mut to_remove_names = Vec::new();

        // System 1: decrement timeout and enqueue entity deletion when expired
        let mut expired = Vec::new();
        {
            let mut q = store.query_mut::<&mut Timeout>().unwrap();
            for (entity, timeout) in q.iter() {
                timeout.value -= dt;
                if timeout.value <= 0.0 {
                    expired.push(entity);
                }
            }
        }
        let mut cmd = store.commands();
        for entity in expired {
            cmd.delete_entity(entity);
            let name = store.name_of(entity);
            elog!("Expire: {} deleted!", name);
            to_remove_names.push(name);
        }

        // System 2: print remaining expiry time (entity is still alive until command buffer flushes)
        for (entity, timeout) in store.query::<&Timeout>().iter() {
            elog!(
                "PrintExpire: {} has {:.2} seconds left",
                store.name_of(entity),
                timeout.value
            );
        }

        // End-of-frame sync point: apply deferred deletions and trigger OnRemove observer
        cmd.apply_discard(&mut store).unwrap();
        for name in to_remove_names {
            elog!("Expired: {} actually deleted", name);
        }

        if !store.is_alive(e) {
            break;
        }

        elog!("Tick...");
    }
}

#[test]
fn test_system_mutate_entity() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "PrintExpire: MyEntity has 1.50 seconds left",
            "Tick...",
            "PrintExpire: MyEntity has 0.50 seconds left",
            "Tick...",
            "Expire: MyEntity deleted!",
            "PrintExpire: MyEntity has -0.50 seconds left",
            "Expired: MyEntity actually deleted",
        ]
    );
}
