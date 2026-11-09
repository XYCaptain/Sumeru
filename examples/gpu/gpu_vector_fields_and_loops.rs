//! Example 9: Vector/Array Fields (`[f32; 3]`, `[u32; 2]`), `for`/`while` Loops & Bitwise Ops.
//!
//! Demonstrates defining `#[gpu]` structs with 3D vector arrays and bitmask arrays,
//! writing kernels with `for` ranges, `while` loops, `continue`, `break`, and bitwise masks,
//! validating the generated WGSL/MSL through `naga`, and executing on GPU.

use crate::common::*;
use crate::elog;

#[gpu]
pub struct Particle3D {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub axis_mask: [u32; 2],
    pub substeps: i32,
}

#[gpu]
pub fn step_particle_3d(p: &mut Particle3D) {
    for axis in 0usize..3usize {
        if (p.axis_mask[0] & (1u32 << (axis as u32))) == 0u32 {
            continue;
        }
        p.pos[axis] += p.vel[axis];
    }
    let mut rem: i32 = p.substeps;
    let mut boost: f32 = 0.0;
    while rem > 0 {
        rem -= 1;
        boost += 0.5;
        if boost >= 1.5 {
            break;
        }
    }
    p.pos[0] += boost;
    p.axis_mask[1] |= 0xFFu32;
}

pub fn main() {
    clear_log();

    let desc = step_particle_3d.descriptor();
    let compiled = compile_and_validate_pipeline(desc).unwrap();

    let msl_def = Particle3D::msl_struct_definition();
    let wgsl_def = Particle3D::wgsl_struct_definition();

    elog!(
        "vec3_shader: validated={}, msl_packed_float3={}, wgsl_array_f32_3={}",
        compiled.shader_validated,
        msl_def.contains("packed_float3 pos;"),
        wgsl_def.contains("pos: array<f32, 3>,")
    );

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);

    // e1 enables X and Z axes (0b101 = 5), skips Y axis via `continue`, runs 5 substeps (capped at boost=1.5 via `break`)
    store
        .insert((
            e1,
            (Particle3D {
                pos: [1.0, 2.0, 3.0],
                vel: [10.0, 20.0, 30.0],
                axis_mask: [5, 0],
                substeps: 5,
            },),
        ))
        .unwrap();

    // e2 enables all 3 axes (0b111 = 7), runs 1 substep (boost=0.5)
    store
        .insert((
            e2,
            (Particle3D {
                pos: [0.0, 0.0, 0.0],
                vel: [2.0, 4.0, 6.0],
                axis_mask: [7, 1],
                substeps: 1,
            },),
        ))
        .unwrap();

    store
        .query_mut::<&mut Particle3D>()
        .unwrap()
        .gpu()
        .for_each(step_particle_3d)
        .unwrap();

    let p1 = *store.get::<Particle3D>(e1).unwrap();
    let p2 = *store.get::<Particle3D>(e2).unwrap();

    elog!(
        "e1_pos=[{:.1}, {:.1}, {:.1}], mask1=0x{:X}",
        p1.pos[0],
        p1.pos[1],
        p1.pos[2],
        p1.axis_mask[1]
    );
    elog!(
        "e2_pos=[{:.1}, {:.1}, {:.1}], mask1=0x{:X}",
        p2.pos[0],
        p2.pos[1],
        p2.pos[2],
        p2.axis_mask[1]
    );
}

#[test]
fn test_gpu_vector_fields_and_loops() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "vec3_shader: validated=true, msl_packed_float3=true, wgsl_array_f32_3=true",
            "e1_pos=[12.5, 2.0, 33.0], mask1=0xFF",
            "e2_pos=[2.5, 4.0, 6.0], mask1=0xFF",
        ]
    );
}
