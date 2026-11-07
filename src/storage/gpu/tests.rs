//! 16KB 硬件页对齐 GPU 零拷贝、Naga 着色器编译校验与三层遍历体系单元测试 (M0 ~ M5)

use crate::storage::column::page_aligned::{PageAlignedBuffer, HARDWARE_PAGE_SIZE};
use crate::storage::gpu::metal::compile_and_validate_pipeline;
use crate::{
    gpu, gpu_component, gpu_kernel, EntityId, GpuKernel, GpuShaderStruct, LayoutOptimization,
    MetalBufferCache, MetalComputeEngine, MetalNoCopyBufferDescriptor, MetalThreadgroupDispatch,
    Store,
};

#[gpu]
pub struct SimPosition {
    pub x: f32,
    pub y: f32,
}

#[gpu]
pub struct SimVelocity {
    pub vx: f32,
    pub vy: f32,
}

#[gpu]
pub struct SimMass {
    pub inv_mass: f32,
}

#[gpu]
pub fn integrate_sim_motion(pos: &mut SimPosition, vel: &SimVelocity) {
    pos.x += vel.vx;
    pos.y += vel.vy;
}

#[gpu]
pub fn damp_velocity(vel: &mut SimVelocity) {
    vel.vx *= 0.5;
    vel.vy *= 0.5;
}

#[gpu]
pub fn integrate_weighted_motion(pos: &mut SimPosition, vel: &SimVelocity, mass: &SimMass) {
    pos.x += vel.vx * mass.inv_mass;
    pos.y += vel.vy * mass.inv_mass;
}

#[gpu]
pub fn clamp_and_early_return(pos: &mut SimPosition, vel: &SimVelocity) {
    let mut step_x: f32 = vel.vx.abs().clamp(0.0, 10.0);
    let scale = (2i32 as f32) * 0.5;
    step_x *= scale;
    pos.x += step_x;
    if pos.x > 100.0 {
        return;
    }
    pos.y += vel.vy;
}

#[gpu_component]
pub struct LegacyAttrPoint {
    pub px: f32,
    pub py: f32,
}

#[gpu_kernel(LegacyPointStepKernel)]
pub fn step_legacy_point(pt: &mut LegacyAttrPoint) {
    pt.px += 1.0;
    pt.py += 2.0;
}

#[test]
fn m0_page_aligned_buffer_16kb_invariants_and_zeroed_tail() {
    let mut buf = PageAlignedBuffer::<SimPosition>::with_capacity(10);
    assert_eq!(buf.as_ptr() as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(buf.allocated_bytes(), HARDWARE_PAGE_SIZE);
    assert_eq!(buf.allocated_page_count(), 1);
    assert_eq!(buf.page_count(), 0);
    let init_gen = buf.generation();

    for i in 0..100 {
        buf.push(SimPosition {
            x: i as f32,
            y: (i * 2) as f32,
        });
    }
    assert_eq!(buf.len(), 100);
    assert_eq!(buf.active_bytes(), 100 * 8);
    assert_eq!(buf.page_count(), 1);
    assert_eq!(buf.generation(), init_gen);

    // 验证 16KB 页内活跃数据尾部均为 0（防止 GPU Threadgroup 尾页越界读取未初始化内存）
    let full_page = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page[buf.active_bytes()..].iter().all(|&b| b == 0));

    // 验证 pop、swap_remove 与 drain 后被腾出的尾部槽位严格归零 (I6)
    let _ = buf.pop();
    assert_eq!(buf.len(), 99);
    let full_page_after_pop = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page_after_pop[buf.active_bytes()..].iter().all(|&b| b == 0));

    let _ = buf.swap_remove(10);
    assert_eq!(buf.len(), 98);
    let full_page_after_swap = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page_after_swap[buf.active_bytes()..].iter().all(|&b| b == 0));

    // 扩容超过 1 个 16KB 页（2048 个 SimPosition = 16384B），触发跨页重分配与代际递增
    for i in 98..2500 {
        buf.push(SimPosition {
            x: i as f32,
            y: i as f32,
        });
    }
    assert_eq!(buf.as_ptr() as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(buf.allocated_bytes() % HARDWARE_PAGE_SIZE, 0);
    assert!(buf.allocated_bytes() >= 2 * HARDWARE_PAGE_SIZE);
    assert_eq!(buf.page_count(), 2);
    assert!(buf.generation() > init_gen);

    let drained = buf.drain(0..2500);
    assert_eq!(drained.len(), 2500);
    assert_eq!(buf.len(), 0);
    let full_after_drain = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_after_drain.iter().all(|&b| b == 0));
}

#[test]
fn m1_zero_copy_gpu_views_dirty_bitmap_and_swap_remove_tracking() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // 插入 2200 个实体，跨越 2 个 16KB 物理页（每页 2048 个 8B 的 SimPosition）
    let mut entities = Vec::with_capacity(2200);
    for i in 0..2200 {
        let e = EntityId::from_u128((i + 1) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: 0.0,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                ),
            ))
            .unwrap();
        entities.push(e);
    }

    let baseline_epoch = store.epoch();
    let views = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    assert_eq!(views.len(), 1);
    let (_, view) = &views[0];
    assert!(view.is_zero_copy_aligned());
    assert_eq!(view.geometry().page_count, 2);
    assert!(!view.dirty_bitmap().has_any_dirty());

    // 修改第 10 行（位于第 0 个 16KB 页），第 1 页应保持干净
    store
        .modify::<SimPosition, _>(entities[10], |p| p.x += 100.0)
        .unwrap();
    let views_after_mod = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    let dirty = views_after_mod[0].1.dirty_bitmap();
    assert!(dirty.is_page_dirty(0));
    assert!(!dirty.is_page_dirty(1));
    let runs = dirty.coalesced_runs();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].first_page, 0);
    assert_eq!(runs[0].byte_offset, 0);
    assert_eq!(runs[0].byte_len, HARDWARE_PAGE_SIZE);

    // 记录新 epoch，然后删除位于第 0 页的实体 entities[20]：
    // swap_remove 会将尾部（第 1 页）实体搬移回填到 row 20（第 0 页），必须标记第 0 页物理脏！
    let pre_delete_epoch = store.epoch();
    store.delete_entity(entities[20]).unwrap();
    let views_after_del = store
        .gpu_column_views::<SimPosition>(pre_delete_epoch)
        .unwrap();
    let dirty_del = views_after_del[0].1.dirty_bitmap();
    assert!(
        dirty_del.is_page_dirty(0),
        "swap_remove must mark destination row's 16KB physical page dirty"
    );
}

#[test]
fn m1_gpu_column_pin_cow_safety_and_page_table() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..128 {
        let e = EntityId::from_u128((i + 100) as u128);
        store
            .insert((
                e,
                (SimPosition {
                    x: i as f32,
                    y: 1.0,
                },),
            ))
            .unwrap();
    }

    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins.len(), 1);
    let pinned_ptr = pins[0].base_ptr();
    assert_eq!(pinned_ptr as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(pins[0].as_slice()[0].x, 0.0);

    // 在 Pin 存活期间对 Store 发起 CPU 修改：触发 COW 复制，Pin 指向的原 16KB 物理页保持不变！
    let first_entity = EntityId::from_u128(100);
    store
        .modify::<SimPosition, _>(first_entity, |p| p.x = 999.0)
        .unwrap();
    assert_eq!(
        pins[0].as_slice()[0].x,
        0.0,
        "Pinned GPU buffer must remain immutable via COW while async GPU ticket holds it"
    );
    assert_eq!(store.get::<SimPosition>(first_entity).unwrap().x, 999.0);

    let page_table = store.build_gpu_page_table::<SimPosition>().unwrap();
    assert_eq!(page_table.total_active_rows(), 128);
    assert_eq!(page_table.total_pages(), 1);
}

#[test]
fn m2_metal_nocopy_descriptor_cache_and_simd32_grid() {
    let grid_empty = MetalThreadgroupDispatch::compute_1d(0, 256);
    assert_eq!(grid_empty.threadgroup_count, 0);

    let grid_small = MetalThreadgroupDispatch::compute_1d(50, 256);
    assert_eq!(grid_small.threads_per_threadgroup, 64); // 向上对齐到 32 的倍数
    assert_eq!(grid_small.threadgroup_count, 1);

    let grid_large = MetalThreadgroupDispatch::compute_1d(1000, 256);
    assert_eq!(grid_large.threads_per_threadgroup, 256);
    assert_eq!(grid_large.threadgroup_count, 4);
    assert_eq!(grid_large.total_grid_threads, 1024);

    // 极端大行数饱和保护（不发生 u32 算术溢出 panic）
    let grid_huge = MetalThreadgroupDispatch::compute_1d(u32::MAX as usize, 256);
    assert!(grid_huge.threadgroup_count > 0);

    // 非 16KB 对齐指针应当被 MetalNoCopyBufferDescriptor 拒绝
    let unaligned_ptr = (HARDWARE_PAGE_SIZE + 64) as *mut u8;
    assert!(
        MetalNoCopyBufferDescriptor::validate_raw(unaligned_ptr, HARDWARE_PAGE_SIZE, 64, 1)
            .is_err()
    );

    let aligned_ptr = HARDWARE_PAGE_SIZE as *mut u8;
    // active_bytes 超过 allocated_bytes 应当被拒绝
    assert!(
        MetalNoCopyBufferDescriptor::validate_raw(
            aligned_ptr,
            HARDWARE_PAGE_SIZE,
            HARDWARE_PAGE_SIZE + 8,
            1
        )
        .is_err()
    );

    let desc =
        MetalNoCopyBufferDescriptor::validate_raw(aligned_ptr, HARDWARE_PAGE_SIZE, 64, 1).unwrap();
    let mut cache = MetalBufferCache::new();
    let arch_id = crate::id::ArchetypeId(1);
    let cid = crate::id::ComponentTypeId::of::<SimPosition>();

    let (h1, hit1) = cache.get_or_insert(arch_id, cid, &desc);
    assert!(!hit1);
    let (h2, hit2) = cache.get_or_insert(arch_id, cid, &desc);
    assert!(hit2);
    assert_eq!(h1.handle_id, h2.handle_id);

    // 当 generation 递增（重分配）时，缓存自动失效并原地替换新句柄（entries.len() 保持为 1）
    let desc_reallocated =
        MetalNoCopyBufferDescriptor::validate_raw(aligned_ptr, HARDWARE_PAGE_SIZE * 2, 64, 2)
            .unwrap();
    let (h3, hit3) = cache.get_or_insert(arch_id, cid, &desc_reallocated);
    assert!(!hit3);
    assert_ne!(h1.handle_id, h3.handle_id);
    assert_eq!(cache.len(), 1);
}

#[test]
fn m3_multiplatform_shader_generation_and_real_naga_compilation() {
    let msl_struct = SimPosition::msl_struct_definition();
    assert!(msl_struct.contains("struct SimPosition"));
    assert!(msl_struct.contains("float x;"));

    let wgsl_struct = SimPosition::wgsl_struct_definition();
    assert!(wgsl_struct.contains("struct SimPosition"));
    assert!(wgsl_struct.contains("x: f32,"));

    let desc = integrate_sim_motion.descriptor();
    assert_eq!(desc.name(), "integrate_sim_motion");
    assert_eq!(desc.bindings().len(), 2);
    assert!(desc.bindings()[0].mutable);
    assert!(!desc.bindings()[1].mutable);

    assert!(desc.msl_source().contains("kernel void integrate_sim_motion"));
    assert!(desc
        .msl_source()
        .contains("device SimPosition* sumeru_buf_pos [[buffer(0)]]"));
    assert!(desc
        .msl_source()
        .contains("const device SimVelocity* sumeru_buf_vel [[buffer(1)]]"));
    assert!(desc
        .msl_source()
        .contains("if (gid >= uniforms.active_rows)"));

    assert!(desc.wgsl_source().contains("@compute @workgroup_size(256)"));
    assert!(desc
        .wgsl_source()
        .contains("var<storage, read_write> sumeru_buf_pos: array<SimPosition>;"));
    assert!(desc
        .wgsl_source()
        .contains("var<storage, read> sumeru_buf_vel: array<SimVelocity>;"));

    // 真实 Naga WGSL 解析 + IR 类型验证 + MSL 后端代码生成校验
    for kernel_desc in [
        integrate_sim_motion.descriptor(),
        damp_velocity.descriptor(),
        integrate_weighted_motion.descriptor(),
        clamp_and_early_return.descriptor(),
        step_legacy_point.descriptor(),
        scale_velocity_with_lifetime.descriptor(),
    ] {
        let compiled = compile_and_validate_pipeline(kernel_desc)
            .expect("Generated WGSL/MSL shader must pass Naga frontend, IR validator, and MSL backend");
        assert!(compiled.shader_validated);
        assert!(compiled.naga_translated_msl.contains("kernel void"));
    }
}

#[test]
fn m3_three_tier_explicit_traversal_cpu_par_and_gpu() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    for i in 0..256 {
        let e = EntityId::from_u128((i + 1) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: (i * 10) as f32,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                    SimMass { inv_mass: 0.5 },
                ),
            ))
            .unwrap();
    }

    // =========================================================================
    // ① CPU 单线程遍历：无类型标注闭包 `.for_each(|(pos, vel)| ...)` 与算子直传 `.run(op)`
    // =========================================================================
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .run(integrate_sim_motion);

    let e0 = EntityId::from_u128(1);
    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 2.0, y: 4.0 }
    );

    // =========================================================================
    // ② CPU 多核并行遍历 (Rayon)：无类型标注闭包 `.par_for_each` 与算子直传 `.par_run` / `.par().run`
    // =========================================================================
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_run(integrate_sim_motion);

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par()
        .run(integrate_sim_motion);

    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 5.0, y: 10.0 }
    );

    // =========================================================================
    // ③ GPU 硬件并行遍历：显式 `.gpu().for_each(op)?` / `.gpu().run(op)?`
    // =========================================================================
    let stats1 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(integrate_sim_motion)
        .unwrap();
    assert_eq!(stats1.dispatched_archetypes, 1);
    assert_eq!(stats1.total_rows, 256);

    let mut engine = MetalComputeEngine::new();
    let stats2 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats2.buffer_cache_misses, 2);
    assert_eq!(stats2.buffer_cache_hits, 0);

    // 第二帧在同一个 MetalComputeEngine 上派发：100% 命中 MTLBuffer 零拷贝缓存！
    let stats3 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .run_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats3.buffer_cache_hits, 2);
    assert_eq!(stats3.buffer_cache_misses, 0);

    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 8.0, y: 16.0 }
    );

    // 单参数与三参数 #[gpu] 算子测试
    store
        .query_mut::<&mut SimVelocity>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, damp_velocity)
        .unwrap();
    assert_eq!(
        *store.get::<SimVelocity>(e0).unwrap(),
        SimVelocity { vx: 0.5, vy: 1.0 }
    );

    store
        .query_mut::<(&mut SimPosition, &SimVelocity, &SimMass)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_weighted_motion)
        .unwrap();
    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 8.25, y: 16.5 }
    );
}

#[gpu]
pub fn scale_velocity_with_lifetime<'a>(vel: &'a mut SimVelocity) {
    vel.vx *= 2.0;
    vel.vy *= 2.0;
}

#[test]
fn m4_completeness_audit_cow_gpu_dispatch_snapshot_merge_and_ro_par() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &mut SimVelocity)>()
        .unwrap();

    for i in 0..64 {
        let e = EntityId::from_u128((i + 500) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: (i * 2) as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 10.0, vy: 20.0 },
                ),
            ))
            .unwrap();
    }

    let mut engine = MetalComputeEngine::new();
    // 1. 首次 GPU 派发缓存 16KB 句柄
    let stats_init = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats_init.buffer_cache_misses, 2);

    // 2. 锁定 SimPosition 列供异步 GPU 票据持有，然后再次执行可变 GPU 派发：
    //    可变列 SimPosition 必须自动触发 COW 独占分离（生成新 16KB 页缓冲与新 generation），
    //    绝不覆盖已 Pin 的旧 16KB 缓冲！而只读列 SimVelocity 则直接命中原零拷贝缓存！
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins[0].as_slice()[0].x, 10.0);

    let stats_cow = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(
        pins[0].as_slice()[0].x,
        10.0,
        "Pinned 16KB buffer must not be overwritten by subsequent GPU mutable dispatch"
    );
    assert_eq!(
        store.get::<SimPosition>(EntityId::from_u128(500)).unwrap().x,
        20.0
    );
    // SimPosition 因 COW 分配了新 generation（1 miss），SimVelocity 保持只读命中（1 hit）
    assert_eq!(stats_cow.buffer_cache_misses, 1);
    assert_eq!(stats_cow.buffer_cache_hits, 1);
    drop(pins);

    // 3. 显式生命周期标注的 #[gpu] 算子测试
    store
        .query_mut::<&mut SimVelocity>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, scale_velocity_with_lifetime)
        .unwrap();
    assert_eq!(
        *store.get::<SimVelocity>(EntityId::from_u128(500)).unwrap(),
        SimVelocity { vx: 20.0, vy: 40.0 }
    );

    // 4. 只读查询 (store.query) 的无类型标注闭包 `.for_each` 与 `.par_for_each` 对称性测试 (C8)
    let mut sum_x = 0.0f32;
    store
        .query::<(&SimPosition, &SimVelocity)>()
        .for_each(|(pos, _vel)| {
            sum_x += pos.x;
        });
    assert!(sum_x > 0.0);

    let par_count = std::sync::atomic::AtomicUsize::new(0);
    store
        .query::<(&SimPosition, &SimVelocity)>()
        .par_for_each(|(pos, vel)| {
            if pos.x >= 0.0 && vel.vx >= 0.0 {
                par_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
    assert_eq!(par_count.load(std::sync::atomic::Ordering::Relaxed), 64);

    // 5. 快照恢复与分区合并后 16KB 页对齐不变量保持验证
    let snap = store.snapshot().unwrap();
    let mut restored_store =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    restored_store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    restored_store.restore_snapshot(&snap).unwrap();
    let restored_views = restored_store.gpu_column_views::<SimPosition>(0).unwrap();
    assert!(restored_views[0].1.is_zero_copy_aligned());

    let mut partition = Store::new(); // 默认 Compact 布局的独立分区
    partition
        .insert((
            EntityId::from_u128(9999),
            (
                SimPosition { x: 1.0, y: 2.0 },
                SimMass { inv_mass: 1.0 },
            ),
        ))
        .unwrap();
    restored_store.merge_partition(partition).unwrap();
    let merged_views = restored_store.gpu_column_views::<SimPosition>(0).unwrap();
    assert_eq!(merged_views.len(), 2);
    assert!(
        merged_views.iter().all(|(_, v)| v.is_zero_copy_aligned()),
        "All archetypes after merge_partition into a PageAlignedGpu store must be 16KB aligned"
    );
}

#[test]
fn m5_mechanism_audit_regressions_c1_c3_c4_c5_c7() {
    // -------------------------------------------------------------------------
    // C1: `gpu_column_views_mut` 必须在 `GpuColumnPin` 存活时触发 COW 独占分离
    // -------------------------------------------------------------------------
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store.register_gpu_component::<SimPosition>().unwrap();
    let e1 = EntityId::from_u128(1);
    store
        .insert((e1, (SimPosition { x: 42.0, y: 7.0 },)))
        .unwrap();

    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins[0].as_slice()[0].x, 42.0);
    let pinned_addr = pins[0].base_ptr() as usize;

    {
        let mut mut_views = store.gpu_column_views_mut::<SimPosition>().unwrap();
        assert_eq!(mut_views.len(), 1);
        let new_addr = mut_views[0].1.base_mut_ptr() as usize;
        assert_ne!(
            new_addr, pinned_addr,
            "gpu_column_views_mut must allocate a fresh COW 16KB buffer when column is pinned"
        );
        mut_views[0].1.as_mut_slice()[0].x = 777.0;
    }
    assert_eq!(
        pins[0].as_slice()[0].x,
        42.0,
        "Pinned buffer must remain untouched after mutation through gpu_column_views_mut"
    );
    assert_eq!(store.get::<SimPosition>(e1).unwrap().x, 777.0);
    drop(pins);

    // -------------------------------------------------------------------------
    // C3: 冷表 (Cold Archetype) 在主 Store 经过多次其他表写入后发生 `swap_remove`，
    //     必须使用当前 Store 的全局 `write_epoch` 标记被覆盖行，不能被增量同步漏掉
    // -------------------------------------------------------------------------
    let mut cold_hot_store =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    cold_hot_store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // 表 A（冷表）：仅含 SimPosition，包含 3 个实体
    let cold_e0 = EntityId::from_u128(10);
    let cold_e1 = EntityId::from_u128(11);
    let cold_e2 = EntityId::from_u128(12);
    for (idx, e) in [cold_e0, cold_e1, cold_e2].into_iter().enumerate() {
        cold_hot_store
            .insert((
                e,
                (SimPosition {
                    x: (idx + 1) as f32,
                    y: 0.0,
                },),
            ))
            .unwrap();
    }

    // 表 B（热表）：含 (SimPosition, SimVelocity)，连续发生 20 次修改推进 Store 全局 epoch
    let hot_e = EntityId::from_u128(20);
    cold_hot_store
        .insert((
            hot_e,
            (
                SimPosition { x: 0.0, y: 0.0 },
                SimVelocity { vx: 1.0, vy: 1.0 },
            ),
        ))
        .unwrap();
    for _ in 0..20 {
        cold_hot_store
            .modify::<SimVelocity, _>(hot_e, |v| v.vx += 1.0)
            .unwrap();
    }

    // 记录删除前的全局快照纪元（此时 global epoch 远大于冷表局部的旧 epoch）
    let before_cold_delete_epoch = cold_hot_store.epoch();

    // 删除冷表第 0 行实体 cold_e0：cold_e2 被 swap_remove 回填到第 0 行
    cold_hot_store.delete_entity(cold_e0).unwrap();

    // 增量脏页视图与 ChangedSince 查询都必须捕获到冷表第 0 页的变更！
    let cold_views = cold_hot_store
        .gpu_column_views::<SimPosition>(before_cold_delete_epoch)
        .unwrap();
    let cold_arch_view = cold_views
        .iter()
        .find(|(_, v)| v.len() == 2)
        .expect("cold archetype must have 2 remaining entities");
    assert!(
        cold_arch_view.1.dirty_bitmap().is_page_dirty(0),
        "Cold archetype swap_remove must stamp global store epoch so dirty page is detected"
    );

    let changed_entities: Vec<EntityId> = cold_hot_store
        .query::<&SimPosition>()
        .changed_since::<SimPosition>(before_cold_delete_epoch)
        .iter()
        .map(|(e, _)| e)
        .collect();
    assert!(
        changed_entities.contains(&cold_e2),
        "ChangedSince must detect entity moved by swap_remove in a cold archetype"
    );

    // -------------------------------------------------------------------------
    // C4 & C5: 空的 PageAlignedGpu 主库先 `register_gpu_query`，再走 `merge_partition` 快路径
    //          合并 Compact 分区，随后执行 `snapshot()` + `restore_snapshot()`
    // -------------------------------------------------------------------------
    let mut empty_target =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    empty_target
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    let mut worker_partition = Store::new();
    for i in 0..32 {
        worker_partition
            .insert((
                EntityId::from_u128((i + 1000) as u128),
                (
                    SimPosition {
                        x: i as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    // 走 merge_disjoint_from 的 empty fast-path
    empty_target.merge_partition(worker_partition).unwrap();
    let fast_merged_views = empty_target.gpu_column_views::<SimPosition>(0).unwrap();
    assert_eq!(fast_merged_views.len(), 1);
    assert!(
        fast_merged_views[0].1.is_zero_copy_aligned(),
        "Fast-path merge into empty PageAlignedGpu store must promote columns to 16KB page alignment"
    );

    // 合并后做 snapshot 与 restore_snapshot 必须成功（Rkyv codec 完整保留）
    let snap = empty_target.snapshot().unwrap();
    let mut restored = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    restored
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    restored.restore_snapshot(&snap).unwrap();
    assert_eq!(restored.len(), 32);
    assert!(
        restored.gpu_column_views::<SimPosition>(0).unwrap()[0]
            .1
            .is_zero_copy_aligned()
    );

    // -------------------------------------------------------------------------
    // C7: 含提前 `return;` 的 `#[gpu]` 内核在 WGSL 与 CPU 中均不得丢失提前返回前的可变写入
    // -------------------------------------------------------------------------
    let early_ret_desc = clamp_and_early_return.descriptor();
    assert!(
        early_ret_desc
            .wgsl_source()
            .contains("if (pos.x > 100.0) { sumeru_buf_pos[gid] = pos; return; }"),
        "WGSL early return must write back mutable buffers before returning: {}",
        early_ret_desc.wgsl_source()
    );

    let mut ret_store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e_high = EntityId::from_u128(9001);
    let e_low = EntityId::from_u128(9002);
    ret_store
        .insert((
            e_high,
            (
                SimPosition { x: 98.0, y: 10.0 },
                SimVelocity { vx: 5.0, vy: 50.0 },
            ),
        ))
        .unwrap();
    ret_store
        .insert((
            e_low,
            (
                SimPosition { x: 10.0, y: 10.0 },
                SimVelocity { vx: 5.0, vy: 50.0 },
            ),
        ))
        .unwrap();

    ret_store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(clamp_and_early_return)
        .unwrap();

    // e_high: pos.x 先加 5.0 变为 103.0 (> 100.0)，触发提前 return，pos.y 保持 10.0，但 pos.x 的修改保留为 103.0！
    assert_eq!(
        *ret_store.get::<SimPosition>(e_high).unwrap(),
        SimPosition { x: 103.0, y: 10.0 }
    );
    // e_low: pos.x 变为 15.0 (<= 100.0)，继续执行 pos.y += 50.0 变为 60.0
    assert_eq!(
        *ret_store.get::<SimPosition>(e_low).unwrap(),
        SimPosition { x: 15.0, y: 60.0 }
    );
}

#[test]
fn m6_runtime_gpu_detection_driver_dispatch_and_morsel_par() {
    use crate::storage::gpu::metal::{
        probe_system_gpu_driver, GpuExecutionMode, GpuHardwareDriver,
    };
    use crate::storage::gpu::{GpuDispatchUniforms, GpuKernelDescriptor};
    use std::sync::{Arc, Mutex};

    // 1. 运行时探测：根据宿主机真实硬件决定 HardwareGpu 或 CpuFallback（绝不硬编码死分支）
    let probed = probe_system_gpu_driver();
    let auto_engine = MetalComputeEngine::new();
    if probed.is_some() {
        assert_eq!(auto_engine.execution_mode(), GpuExecutionMode::HardwareGpu);
        assert!(auto_engine.has_hardware_gpu());
    } else {
        assert_eq!(auto_engine.execution_mode(), GpuExecutionMode::CpuFallback);
        assert!(!auto_engine.has_hardware_gpu());
    }

    // 2. 硬件驱动分支验证：注入自定义 GpuHardwareDriver，验证 dispatch_archetype_raw 走真实驱动路径，
    //    且当缓存发生 COW 世代替换或驱逐时自动调用 unregister_buffer 回收旧 GPU 句柄。
    #[derive(Debug, Default)]
    struct RecordedDriverState {
        compiled_kernels: Vec<String>,
        registered_handles: Vec<u64>,
        unregistered_handles: Vec<u64>,
        dispatched_rows: Vec<usize>,
    }

    #[derive(Debug)]
    struct MockHardwareGpuDriver {
        state: Arc<Mutex<RecordedDriverState>>,
    }

    impl GpuHardwareDriver for MockHardwareGpuDriver {
        fn device_name(&self) -> &str {
            "Mock-UMA-Hardware-GPU"
        }

        fn has_unified_memory(&self) -> bool {
            true
        }

        fn compile_pipeline(
            &mut self,
            _descriptor: &GpuKernelDescriptor,
            compiled: &crate::MetalCompiledPipeline,
        ) -> Result<u32, crate::StateError> {
            self.state
                .lock()
                .unwrap()
                .compiled_kernels
                .push(compiled.entry_point.to_string());
            Ok(256)
        }

        unsafe fn register_no_copy_buffer(
            &mut self,
            handle_id: u64,
            desc: &MetalNoCopyBufferDescriptor,
        ) -> Result<(), crate::StateError> {
            assert!(
                (desc.base_ptr as usize) % HARDWARE_PAGE_SIZE == 0
                    && desc.allocated_bytes % HARDWARE_PAGE_SIZE == 0,
                "Hardware driver must only receive 16KB page-aligned zero-copy buffers"
            );
            self.state
                .lock()
                .unwrap()
                .registered_handles
                .push(handle_id);
            Ok(())
        }

        fn unregister_buffer(&mut self, handle_id: u64) {
            self.state
                .lock()
                .unwrap()
                .unregistered_handles
                .push(handle_id);
        }

        unsafe fn dispatch_compute(
            &mut self,
            descriptor: &GpuKernelDescriptor,
            buffer_handles: &[u64],
            raw_ptrs: &[*mut u8],
            active_rows: usize,
            _grid: &MetalThreadgroupDispatch,
            uniforms: &GpuDispatchUniforms,
        ) -> Result<(), crate::StateError> {
            assert_eq!(buffer_handles.len(), raw_ptrs.len());
            self.state
                .lock()
                .unwrap()
                .dispatched_rows
                .push(active_rows);
            // 模拟 GPU 硬件在 UMA 零拷贝页上直接执行着色器逻辑
            unsafe {
                (descriptor.cpu_runner())(raw_ptrs, active_rows, uniforms);
            }
            Ok(())
        }
    }

    let recorded = Arc::new(Mutex::new(RecordedDriverState::default()));
    let mut hw_engine = MetalComputeEngine::with_driver(Box::new(MockHardwareGpuDriver {
        state: Arc::clone(&recorded),
    }));
    assert_eq!(hw_engine.execution_mode(), GpuExecutionMode::HardwareGpu);
    assert_eq!(hw_engine.device_name(), "Mock-UMA-Hardware-GPU");

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    for i in 0..128u128 {
        store
            .insert((
                EntityId::from_u128(i + 1),
                (
                    SimPosition {
                        x: i as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    let stats1 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut hw_engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats1.execution_mode, GpuExecutionMode::HardwareGpu);
    assert_eq!(stats1.total_rows, 128);
    assert_eq!(stats1.buffer_cache_misses, 2);
    assert_eq!(
        *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap(),
        SimPosition { x: 2.0, y: 4.0 }
    );

    // 持有零拷贝列锁定票据 (pin_gpu_columns) 期间发起第二次 GPU 可变写入：
    // Phase 1b 触发 Arc::make_mut COW 克隆并自增 buffer_generation，
    // MetalBufferCache 按 (archetype_id, component_id) 替换旧条目并通知硬件驱动 unregister_buffer
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    let stats2 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut hw_engine, integrate_sim_motion)
        .unwrap();
    drop(pins);
    assert_eq!(stats2.execution_mode, GpuExecutionMode::HardwareGpu);
    assert_eq!(stats2.buffer_cache_hits, 1); // 只读列 SimVelocity 保持原世代命中
    assert_eq!(stats2.buffer_cache_misses, 1); // 可写列 SimPosition 因 COW 世代递增重新注册

    {
        let guard = recorded.lock().unwrap();
        assert_eq!(guard.compiled_kernels, vec!["integrate_sim_motion"]);
        assert_eq!(guard.dispatched_rows, vec![128, 128]);
        assert_eq!(guard.registered_handles.len(), 3);
        assert_eq!(
            guard.unregistered_handles.len(),
            1,
            "COW generation bump must unregister stale MTLBuffer handle from hardware driver"
        );
    }

    // 3. Morsel-Driven 零物化并行迭代验证（> 2500 行跨多个 1024 行 Morsel 分片）
    const ACTIVE_FLAG: u8 = 1;
    let mut par_store = Store::new();
    for i in 0..2560u128 {
        let e = EntityId::from_u128(i + 10_000);
        par_store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: (i * 2) as f32,
                    },
                    SimVelocity { vx: 1.5, vy: -0.5 },
                ),
            ))
            .unwrap();
        if i % 2 == 0 {
            par_store.set_flag::<ACTIVE_FLAG>(e, true).unwrap();
        }
    }

    // 3a. 带 with_flag 过滤的跨 Morsel `par_for_each` 可变并行更新
    par_store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .with_flag::<ACTIVE_FLAG>()
        .par_for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    // 验证偶数行（有 ACTIVE_FLAG）已更新，奇数行未变
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(10_000))
            .unwrap(),
        SimPosition { x: 1.5, y: -0.5 }
    );
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(10_001))
            .unwrap(),
        SimPosition { x: 1.0, y: 2.0 }
    );
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(12_558))
            .unwrap(),
        SimPosition {
            x: 2558.0 + 1.5,
            y: 5116.0 - 0.5
        }
    );

    // 3b. 只读 SelectQuery 跨 Morsel `par().for_each` 并行聚合
    use std::sync::atomic::{AtomicUsize, Ordering};
    let visited = AtomicUsize::new(0);
    par_store
        .query::<(&SimPosition, &SimVelocity)>()
        .with_flag::<ACTIVE_FLAG>()
        .par()
        .for_each(|(_pos, _vel)| {
            visited.fetch_add(1, Ordering::Relaxed);
        });
    assert_eq!(visited.load(Ordering::Relaxed), 1280);

    // 4. M7-1 & M7-2 机制完备性回归：
    //    - M7-1: 不匹配查询 Q 的其它 Archetype（例如仅含 SimPosition 不含 SimVelocity）绝不能在 query_mut 遍历中被误标脏页；
    //    - M7-2: 被 pin_gpu_columns 锁定的只读列（SimVelocity）在 par_for_each 中绝不触发多余的 COW 克隆。
    let pos_only_entity = EntityId::from_u128(999_001);
    store
        .insert((pos_only_entity, (SimPosition { x: 42.0, y: 42.0 },)))
        .unwrap();
    let epoch_before_mut = store.epoch();
    let vel_pins = store.pin_gpu_columns::<SimVelocity>().unwrap();
    let vel_gen_before = vel_pins[0].generation();

    // CPU 顺序 query_mut::<(&mut SimPosition, &SimVelocity)> 与并行 par_for_each
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .for_each(|(pos, vel)| {
            pos.x += vel.vx;
        });
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_for_each(|(pos, vel)| {
            pos.y += vel.vy;
        });

    // 验证仅含 SimPosition 的不匹配 Archetype 未被误标脏
    let pos_views = store.gpu_column_views::<SimPosition>(epoch_before_mut).unwrap();
    let pos_only_view = pos_views
        .iter()
        .find(|(_, v)| v.len() == 1)
        .expect("pos-only archetype must exist");
    assert!(
        !pos_only_view.1.dirty_bitmap().has_any_dirty(),
        "Unmatched archetype must not be marked dirty by query_mut::<(&mut SimPosition, &SimVelocity)>"
    );

    // 验证只读列 SimVelocity 在持有 Pin 期间执行 par_for_each 未触发 COW 重分配（generation 不变）
    let vel_views = store.gpu_column_views::<SimVelocity>(0).unwrap();
    assert_eq!(
        vel_views[0].1.generation(),
        vel_gen_before,
        "Read-only column in par_for_each must not trigger COW clone while pinned"
    );
    drop(vel_pins);

    // 5. M8-1..M8-5 深度机制回归：
    //    - M8-1: restore_snapshot_owned 必须与 restore_snapshot 一致，推进并烙印 next_epoch 且归一化 16KB 页对齐；
    //    - M8-3 & M8-4: merge_disjoint_from 合并持有 GpuColumnPin 的源分区时不得触发源列多余 COW，且自动升级目标 Store 的 16KB 页对齐原型；
    //    - M8-5: Store::clone 将 epoch 重置为基线 1 时，Archetype 块纪元同步重置为 1，确保 changed_since(cloned.epoch()) 不会误报历史块。
    let snap_owned = store.snapshot().unwrap();
    let pre_owned_restore_epoch = store.epoch();
    store.restore_snapshot_owned(snap_owned).unwrap();
    assert!(
        store.epoch() > pre_owned_restore_epoch,
        "restore_snapshot_owned must advance epoch"
    );
    let owned_restored_views = store
        .gpu_column_views::<SimPosition>(pre_owned_restore_epoch)
        .unwrap();
    assert!(
        owned_restored_views
            .iter()
            .all(|(_, v)| v.is_zero_copy_aligned() && v.dirty_bitmap().has_any_dirty()),
        "restore_snapshot_owned must preserve 16KB page alignment and stamp dirty epoch > pre_owned_restore_epoch"
    );

    let mut cloned_store = store.clone();
    assert_eq!(cloned_store.epoch(), 1);
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(cloned_store.epoch())
            .iter()
            .count(),
        0,
        "Freshly cloned Store must not report pre-clone rows as changed_since(cloned.epoch())"
    );
    cloned_store
        .get_mut::<SimPosition>(EntityId::from_u128(1))
        .unwrap()
        .x += 1.0;
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(1)
            .iter()
            .count(),
        64,
        "Cloned Store mutation on row 0 must only mark chunk 0 (64 rows out of 128) as changed_since(1)"
    );

    // 6. M9-1..M9-4 深度机制回归：
    //    - M9-2: update_batch 仅修改第 0 块（row 0）时，绝不能把第 1 块（rows 64..128）误标脏；
    //    - M9-3: CommandBuffer 原位单组件/多组件更新仅修改第 0 块时，绝不能把第 1 块误标脏；
    //    - M9-1: EntityBundle 快路径插入已有 Archetype 时必须烙印 next_epoch，使 changed_since 精确捕获新行。
    let epoch_before_direct = cloned_store.epoch();
    cloned_store
        .update_batch_discard(std::iter::once((
            EntityId::from_u128(1),
            SimPosition { x: 77.0, y: 88.0 },
        )))
        .unwrap();
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_direct)
            .iter()
            .count(),
        64,
        "update_batch_discard on row 0 must only mark chunk 0 (64 rows out of 128) dirty"
    );

    let epoch_before_cmd = cloned_store.epoch();
    let mut cmd = cloned_store.commands();
    cmd.update_batch(std::iter::once((
        EntityId::from_u128(1),
        SimPosition { x: 99.0, y: 111.0 },
    )));
    cmd.apply_discard(&mut cloned_store).unwrap();
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_cmd)
            .iter()
            .count(),
        64,
        "CommandBuffer native update on row 0 must only mark chunk 0 (64 rows out of 128) dirty"
    );

    let epoch_before_bundle = cloned_store.epoch();
    let mut bundle = crate::EntityBundle::new();
    bundle.insert(SimPosition { x: 5.0, y: 6.0 });
    cloned_store
        .create_entity(EntityId::from_u128(999_002), bundle)
        .unwrap();
    assert!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_bundle)
            .iter()
            .any(|(e, _)| e == EntityId::from_u128(999_002)),
        "EntityBundle fast-path insertion into existing archetype must stamp next_epoch"
    );

    // 7. M10-1..M10-6 深度机制回归：
    //    - M10-1: StateSpace (WorldEngine) 的 for_each_chunk_mut* / par_for_each_chunk_mut* 必须同步烙印写入列的块纪元；
    //    - M10-2: Store::modify_tuple 仅标记元组内被借用修改的列，不误标同表内未触碰的其它列；
    //    - M10-4: chunks_mut() 配合行级过滤器返回整表连续切片时，全表所有 64 行块均须标记写纪元，且只读 chunks() 跳过零匹配表；
    //    - M10-5: merge_disjoint_from 向非空 Store 合并时（无论追加至已有原型或新建原型）必须完整保留非迁移行级状态位平面 (flag_planes)；
    //    - M10-6: try_get_mut / get_mut 在缺失组件失败时绝不污染块纪元。
    {
        use crate::al::{All as AlAll, StateSpace};
        let ep_before_al = cloned_store.epoch();
        StateSpace::for_each_chunk_mut2::<SimPosition, SimVelocity, AlAll>(
            &mut cloned_store,
            |_ents, pos, vel| {
                for (p, v) in pos.iter_mut().zip(vel.iter()) {
                    p.x += v.vx;
                }
            },
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimPosition>(ep_before_al)
                .iter()
                .count(),
            128,
            "M10-1: StateSpace::for_each_chunk_mut2 must mark written column SimPosition chunk epochs"
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimVelocity>(ep_before_al)
                .iter()
                .count(),
            0,
            "M10-1: StateSpace::for_each_chunk_mut2 must not mark read-only column SimVelocity dirty"
        );

        let ep_before_al_par = cloned_store.epoch();
        StateSpace::par_for_each_chunk_mut2::<SimPosition, SimVelocity, AlAll>(
            &mut cloned_store,
            |_ents, pos, vel| {
                for (p, v) in pos.iter_mut().zip(vel.iter()) {
                    p.y += v.vy;
                }
            },
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimPosition>(ep_before_al_par)
                .iter()
                .count(),
            128,
            "M10-1: StateSpace::par_for_each_chunk_mut2 must mark written column SimPosition chunk epochs"
        );
    }

    // M10-2: modify_tuple column isolation
    let ep_before_mod_tuple = cloned_store.epoch();
    cloned_store
        .modify_tuple::<(SimPosition,), _>(EntityId::from_u128(1), |(pos,)| {
            pos.x += 10.0;
        })
        .unwrap();
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimPosition>(ep_before_mod_tuple)
            .iter()
            .count(),
        64,
        "M10-2: modify_tuple::<(SimPosition,)> on row 0 must mark SimPosition chunk 0 dirty"
    );
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimVelocity>(ep_before_mod_tuple)
            .iter()
            .count(),
        0,
        "M10-2: modify_tuple::<(SimPosition,)> must NOT mark untouched column SimVelocity dirty"
    );

    // M10-4: chunks() & chunks_mut() with row-level filter
    let ep_after_row0_mod = ep_before_mod_tuple; // only chunk 0 of SimPosition in (SimPosition, SimVelocity) is dirty since ep_before_mod_tuple
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>( cloned_store.epoch())
            .chunks()
            .count(),
        0,
        "M10-4: read-only chunks() must skip archetypes where row filter matches 0 rows"
    );
    let ep_before_chunks_mut = cloned_store.epoch();
    for (_ents, pos_slice) in cloned_store
        .query_mut::<&mut SimPosition>()
        .unwrap()
        .changed_since::<SimPosition>(ep_after_row0_mod)
        .chunks_mut()
    {
        // Mutate a row in chunk 1 (row 100) of the yielded whole-archetype slice
        if pos_slice.len() > 100 {
            pos_slice[100].x += 5.0;
        }
    }
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimPosition>(ep_before_chunks_mut)
            .iter()
            .count(),
        128,
        "M10-4: chunks_mut() yields the entire archetype slice and must mark all chunks of that archetype dirty"
    );

    // M10-5: merge_disjoint_from into non-empty Store preserves flag_planes
    let mut flag_src = Store::new();
    let e_existing_arch = EntityId::from_u128(888_001);
    let e_new_arch = EntityId::from_u128(888_002);
    flag_src
        .insert((
            e_existing_arch,
            (SimPosition { x: 1.0, y: 2.0 }, SimVelocity { vx: 0.1, vy: 0.2 }),
        ))
        .unwrap();
    flag_src
        .insert((e_new_arch, (SimVelocity { vx: 9.0, vy: 9.0 },)))
        .unwrap();
    flag_src.set_flag::<7>(e_existing_arch, true).unwrap();
    flag_src.set_flag::<7>(e_new_arch, true).unwrap();
    cloned_store.merge_disjoint_from(flag_src).unwrap();
    assert!(
        cloned_store.has_flag::<7>(e_existing_arch),
        "M10-5: merge_disjoint_from must preserve row flags when appending to existing archetype"
    );
    assert!(
        cloned_store.has_flag::<7>(e_new_arch),
        "M10-5: merge_disjoint_from must preserve row flags when moving new archetype"
    );

    // M10-6: failed try_get_mut does not advance epoch or stamp chunk epochs
    let ep_before_failed_get_mut = cloned_store.epoch();
    assert!(
        cloned_store
            .try_get_mut::<SimVelocity>(pos_only_entity)
            .is_err()
    );
    assert_eq!(cloned_store.epoch(), ep_before_failed_get_mut);
}


