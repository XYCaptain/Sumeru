//! Port of `examples/flecs/systems/system_time_interval.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Timeout {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct IntervalTimer {
    pub interval: f32,
    pub elapsed: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let timeout_entity = store.spawn_auto().unwrap();
    store.set(timeout_entity, Timeout { value: 3.5 }).unwrap();

    let tick_sys = store.entity_named("Tick");
    store
        .set(
            tick_sys,
            IntervalTimer {
                interval: 1.0,
                elapsed: 0.0,
            },
        )
        .unwrap();

    let fast_tick_sys = store.entity_named("FastTick");
    store
        .set(
            fast_tick_sys,
            IntervalTimer {
                interval: 0.5,
                elapsed: 0.0,
            },
        )
        .unwrap();

    // Simulate 0.5s ticks until Timeout <= 0.0
    let dt = 0.5_f32;
    loop {
        // System 1: advance timeout
        {
            let mut q = store.query_mut::<&mut Timeout>().unwrap();
            for (_e, t) in q.iter() {
                t.value -= dt;
            }
        }

        // System 2 & 3: advance interval timers in Store and fire when elapsed >= interval
        let mut fired = Vec::new();
        {
            let mut q = store.query_mut::<&mut IntervalTimer>().unwrap();
            for (e, timer) in q.iter() {
                timer.elapsed += dt;
                if timer.elapsed + 1e-5 >= timer.interval {
                    timer.elapsed -= timer.interval;
                    fired.push(e);
                }
            }
        }
        // Order FastTick before Tick to match Flecs interval timer output
        fired.sort_by_key(|&e| if e == fast_tick_sys { 0 } else { 1 });
        for e in fired {
            elog!("{}", store.name_of(e));
        }

        if store.get::<Timeout>(timeout_entity).unwrap().value <= 0.0 {
            elog!("Timed out!");
            break;
        }
    }
}

#[test]
fn test_system_time_interval() {
    main();
    let lines = take_log();
    assert_eq!(lines.last().map(String::as_str), Some("Timed out!"));
    assert!(lines.contains(&"Tick".to_string()));
    assert!(lines.contains(&"FastTick".to_string()));
}
