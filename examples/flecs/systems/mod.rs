pub mod system_basics;
pub mod system_ctx;
pub mod system_custom_phases;
pub mod system_custom_phases_no_builtin;
pub mod system_custom_pipeline;
pub mod system_custom_runner;
pub mod system_delta_time;
pub mod system_immediate;
pub mod system_mutate_entity;
pub mod system_mutate_entity_handle;
pub mod system_pipeline;
pub mod system_startup_system;
pub mod system_sync_point;
pub mod system_sync_point_delete;
pub mod system_target_fps;
pub mod system_time_interval;

pub fn run_all() {
    system_basics::main();
    system_ctx::main();
    system_custom_phases::main();
    system_custom_phases_no_builtin::main();
    system_custom_pipeline::main();
    system_custom_runner::main();
    system_delta_time::main();
    system_immediate::main();
    system_mutate_entity::main();
    system_mutate_entity_handle::main();
    system_pipeline::main();
    system_startup_system::main();
    system_sync_point::main();
    system_sync_point_delete::main();
    system_target_fps::main();
    system_time_interval::main();
}
