pub mod observer_basics;
pub mod observer_custom_event;
pub mod observer_entity_event;
pub mod observer_monitor;
pub mod observer_propagate;
pub mod observer_two_components;
pub mod observer_yield_existing;

pub fn run_all() {
    observer_basics::main();
    observer_custom_event::main();
    observer_entity_event::main();
    observer_monitor::main();
    observer_propagate::main();
    observer_two_components::main();
    observer_yield_existing::main();
}
