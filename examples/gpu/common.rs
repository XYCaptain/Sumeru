//! Shared logging harness and common `#[gpu]` simulation components for `examples/gpu`.

use std::cell::RefCell;

pub use sumeru::al::Tag;
pub use sumeru::{
    compile_and_validate_pipeline, gpu, ArchetypeId, EntityId, GpuComponent, GpuDispatchStats,
    GpuExecutionMode, GpuKernel, GpuShaderStruct, LayoutOptimization, MetalAsyncCommandTicket,
    MetalBufferCache, MetalComputeEngine, MetalNoCopyBufferDescriptor, MetalThreadgroupDispatch,
    PageAlignedBuffer, Query, QueryBuilder, StateError, Store,
};

thread_local! {
    static GPU_EXAMPLE_LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Clears the thread-local example output buffer before running an example.
pub fn clear_log() {
    GPU_EXAMPLE_LOG.with(|log| log.borrow_mut().clear());
}

/// Appends a formatted line to both stdout and the thread-local test log.
pub fn log_line(line: impl Into<String>) {
    let s = line.into();
    println!("{s}");
    GPU_EXAMPLE_LOG.with(|log| log.borrow_mut().push(s));
}

/// Returns all lines logged during the current example execution.
pub fn take_log() -> Vec<String> {
    GPU_EXAMPLE_LOG.with(|log| log.borrow().clone())
}

#[macro_export]
macro_rules! elog {
    () => {
        $crate::common::log_line("")
    };
    ($($arg:tt)*) => {
        $crate::common::log_line(format!($($arg)*))
    };
}

/// 2D position component aligned for 16KB zero-copy GPU dispatch (`8` bytes per row).
#[gpu]
pub struct SimPosition {
    pub x: f32,
    pub y: f32,
}

/// 2D velocity component aligned for 16KB zero-copy GPU dispatch (`8` bytes per row).
#[gpu]
pub struct SimVelocity {
    pub vx: f32,
    pub vy: f32,
}

/// Inverse-mass component for weighted force integration (`4` bytes per row).
#[gpu]
pub struct SimMass {
    pub inv_mass: f32,
}

/// Linear drag / damping factor component (`4` bytes per row).
#[gpu]
pub struct SimDrag {
    pub factor: f32,
}

/// Basic Euler motion integration kernel compiled simultaneously to CPU, Apple MSL, and WebGPU WGSL.
#[gpu]
pub fn integrate_motion(pos: &mut SimPosition, vel: &SimVelocity) {
    pos.x += vel.vx;
    pos.y += vel.vy;
}
