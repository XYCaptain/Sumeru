//! Unified runner and test harness for all 13 Sumeru 16KB Page-Aligned Zero-Copy GPU Compute examples.
//!
//! Run all GPU examples:
//!   `cargo run --example gpu --features gpu`
//!
//! Run all GPU example tests:
//!   `cargo test --example gpu --features gpu`

#![allow(dead_code, unused_imports, unused_mut, unused_must_use)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[macro_use]
pub mod common;

pub mod gpu_archetype_filtering_and_multi_table;
pub mod gpu_cow_pin_and_async_safety;
pub mod gpu_cross_tier_numerical_parity;
pub mod gpu_engine_buffer_cache_and_grid;
pub mod gpu_hello_world;
pub mod gpu_page_aligned_dirty_tracking;
pub mod gpu_partition_merge_and_snapshot;
pub mod gpu_persistent_query_and_views_mut;
pub mod gpu_precheck_and_auto_promotion;
pub mod gpu_shader_pipeline_inspection;
pub mod gpu_three_tier_execution;
pub mod gpu_typed_ir_consts_and_helpers;
pub mod gpu_vector_fields_and_loops;

fn main() {
    println!("=== Running all 13 GPU zero-copy compute examples on sumeru::Store ===");
    gpu_hello_world::main();
    gpu_three_tier_execution::main();
    gpu_shader_pipeline_inspection::main();
    gpu_page_aligned_dirty_tracking::main();
    gpu_cow_pin_and_async_safety::main();
    gpu_engine_buffer_cache_and_grid::main();
    gpu_archetype_filtering_and_multi_table::main();
    gpu_partition_merge_and_snapshot::main();
    gpu_vector_fields_and_loops::main();
    gpu_cross_tier_numerical_parity::main();
    gpu_precheck_and_auto_promotion::main();
    gpu_persistent_query_and_views_mut::main();
    gpu_typed_ir_consts_and_helpers::main();
    println!("=== Completed all 13 GPU zero-copy compute examples on sumeru::Store ===");
}
