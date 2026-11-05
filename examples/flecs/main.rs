//! Unified runner and test harness for all 85 Flecs-canonical examples implemented on `sumer_state::Store`.
//!
//! Run all examples:
//!   `cargo run -p sumer_state --example flecs_examples`
//!
//! Run all 85 example tests:
//!   `cargo test -p sumer_state --example flecs_examples`

#![allow(dead_code, unused_imports, unused_mut, unused_must_use)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[macro_use]
pub mod common;

pub mod entities;
pub mod game_mechanics;
pub mod hello_world;
pub mod observers;
pub mod prefabs;
pub mod queries;
pub mod reflection;
pub mod relationships;
pub mod systems;

fn main() {
    println!("=== Running all 85 Flecs examples on sumer_state::Store ===");
    hello_world::main();
    entities::run_all();
    game_mechanics::run_all();
    relationships::run_all();
    prefabs::run_all();
    queries::run_all();
    systems::run_all();
    observers::run_all();
    reflection::run_all();
    println!("=== Completed all 85 Flecs examples on sumer_state::Store ===");
}
