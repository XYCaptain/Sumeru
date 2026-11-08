//! 异构计算与统一内存/脏页增量同步子系统 (Heterogeneous GPU Compute & Dirty-Page Fabric)
//!
//! 核心能力：
//! 1. **UMA 16KB 硬件页对齐与零填充尾页**：基于 [`HARDWARE_PAGE_SIZE`]（16KB，同时支持 4KB PCIe 与
//!    64KB NVIDIA Grace Hopper NVLink-C2C 系统页参数化）构建零拷贝列视图；
//! 2. **64 行物理块纪元到硬件页的脏追踪折叠**：[`PageDirtyBitmap::from_phys_chunk_epochs`] 将
//!    `Archetype` 内部 64 行粒度的块变更纪元精确折叠为连续硬件脏页段 [`DirtyPageRun`]，
//!    将 PCIe / NVLink 跨总线 DMA 传输严格收敛在被触碰的物理页内；
//! 3. **COW 零阻塞冻结保活 (`GpuColumnPin`)**：利用 [`ColumnBlock`] 的引用计数写时复制机制，
//!    主循环可在纳秒级冻结当前时间步列数据并移交 GPU 或异步 Checkpointing 通道；
//! 4. **打破单行隔离的异构计算流水线 (`GpuComputePass` & `KernelIr`)**：支持 `RowFlags` 位平面谓词掩码、
//!    `Pair<R, Target>` 跨行关系目标 Gather / Scatter-Reduce、双精度 `F64` 归约以及
//!    WGSL / Metal (MSL) / CUDA C++ 三后端代码生成。

pub mod ir;
pub mod metal;
pub mod pass;

pub use ir::{
    AtomicOp, BinaryOp, BufferAccess, BufferBinding, GpuScalarKind, IrExpr, IrStmt, KernelIr,
    UnaryOp,
};
pub use metal::{GpuPredicationMaskBuffer, MetalNoCopyDescriptor, PairTargetGpuLut};
pub use pass::{ArchetypeDispatchStats, GpuComputePass, GpuDispatchReceipt};

use crate::context::ModelComponent;
use crate::id::ComponentTypeId;
use crate::storage::core::archetype::RowBitMask;
use crate::storage::core::column::{ColumnBlock, PageAlignedBuffer};
use crate::storage::core::store::Store;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::sync::Arc;

/// 传统 x86_64 / PCIe 独显常用 4KB 系统页粒度
pub const PCIE_4K_PAGE_SIZE: usize = 4096;

use crate::storage::primitives::{
    DirtyPageRun, PageDirtyBitmap, GRACE_HOPPER_64K_PAGE_SIZE, HARDWARE_PAGE_SIZE,
};

/// GPU 间接调度参数（对齐 WebGPU `dispatchWorkgroupsIndirect` / Metal / Vulkan / CUDA 标准 3D 网格）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectDispatchArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

impl IndirectDispatchArgs {
    #[inline]
    pub const fn from_row_count(row_count: usize, workgroup_size: u32) -> Self {
        let wg = if workgroup_size == 0 { 64 } else { workgroup_size };
        let groups = (row_count as u32).div_ceil(wg);
        Self {
            workgroups_x: groups,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    }

    /// 根据 `RowBitMask` 的最大有效置位跨度计算紧凑间接调度网格
    #[inline]
    pub fn from_active_mask(mask: &RowBitMask, workgroup_size: u32) -> Self {
        if mask.is_all_zero() {
            return Self {
                workgroups_x: 0,
                workgroups_y: 0,
                workgroups_z: 0,
            };
        }
        Self::from_row_count(mask.row_len(), workgroup_size)
    }
}

/// 零阻塞 COW 列内存保活句柄 (`GpuColumnPin<T>`)
///
/// 利用 [`ColumnBlock<T>`] 的 `Arc` 写时复制机制：
/// - 在主循环调用 [`Store::pin_column_for_gpu`] 时，仅递增引用计数（纳秒级，零数据拷贝）；
/// - 若下一时间步 CPU 修改了该列，`DirectBlock::Frozen` 会自动触发写时复制（COW）分离新缓冲，
///   而 GPU 计算通道或后台 Checkpointing 线程持有的 `GpuColumnPin<T>` 仍安全指向冻结时刻的物理内存。
#[derive(Debug, Clone)]
pub struct GpuColumnPin<T> {
    archetype_id: u32,
    pinned_epoch: u64,
    block: Arc<ColumnBlock<T>>,
}

impl<T> GpuColumnPin<T> {
    #[inline]
    pub fn archetype_id(&self) -> u32 {
        self.archetype_id
    }

    #[inline]
    pub fn pinned_epoch(&self) -> u64 {
        self.pinned_epoch
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.block.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.block.as_slice()
    }

    #[inline]
    pub fn arc_block(&self) -> &Arc<ColumnBlock<T>> {
        &self.block
    }

    /// 导出 16KB 硬件页对齐且尾页零填充的 [`PageAlignedBuffer<T>`]（供 UMA `newBufferWithBytesNoCopy` 绑定）
    pub fn to_page_aligned_buffer(&self) -> PageAlignedBuffer<T>
    where
        T: Clone,
    {
        PageAlignedBuffer::from_slice_zero_padded(self.block.as_slice())
    }
}

impl Store {
    /// 零拷贝 COW 冻结并钉住指定原型表的组件列 `T`，返回 [`GpuColumnPin<T>`]。
    ///
    /// 冻结后，主计算循环可立即进入下一时间步修改该列（触发自动 COW），不会阻塞或破坏已钉住的列视图。
    pub fn pin_column_for_gpu<T: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &mut self,
        archetype_id: u32,
    ) -> Option<GpuColumnPin<T>> {
        let pinned_epoch = self.epoch;
        let arch = self.archetypes.get_mut(archetype_id as usize)?;
        let col = arch.get_column_mut::<T>()?;
        let block = col.backing_mut().freeze();
        Some(GpuColumnPin {
            archetype_id,
            pinned_epoch,
            block,
        })
    }

    /// 计算指定原型表的组件列 `T` 自 `since_epoch` 以来的 16KB 硬件脏页位图与连续传输段
    pub fn column_dirty_pages<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
        archetype_id: u32,
        since_epoch: u64,
    ) -> Option<PageDirtyBitmap> {
        let arch = self.archetypes.get(archetype_id as usize)?;
        let cid = ComponentTypeId::of::<T>();
        let (epochs, def_ep) = arch.column_chunk_epochs(cid)?;
        Some(PageDirtyBitmap::from_phys_chunk_epochs(
            epochs,
            def_ep,
            since_epoch,
            arch.len(),
            std::mem::size_of::<T>(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::EntityId;
    use crate::storage::query::QueryBuilder;
    use serde::Deserialize;

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct Pos3D {
        x: f64,
        y: f64,
        z: f64,
        w: f64, // 32 bytes per element -> 512 elements per 16KB hardware page (8 chunks of 64 rows)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct Vel3D {
        vx: f64,
        vy: f64,
        vz: f64,
        vw: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct ForceAccum {
        fx: f64,
        fy: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct AttachedTo;

    #[test]
    fn test_64row_chunk_epochs_fold_to_16kb_dirty_page_runs() {
        // Pos3D = 32 bytes.
        // 1 hardware page (16,384 bytes) holds 512 rows = 8 physical chunks of 64 rows (chunks 0..8).
        // Total 2048 rows = 4 pages (pages 0, 1, 2, 3), 32 chunks (chunks 0..32).
        let row_len = 2048;
        let elem_size = std::mem::size_of::<Pos3D>();
        assert_eq!(elem_size, 32);

        let mut chunk_epochs = vec![10u64; 32];
        // Touch chunk 2 (in page 0) and chunk 5 (also in page 0) -> folds into a single dirty page 0!
        chunk_epochs[2] = 15;
        chunk_epochs[5] = 15;
        // Touch chunk 24 (in page 3) -> page 3 is dirty, while pages 1 and 2 remain clean!
        chunk_epochs[24] = 16;

        let bitmap =
            PageDirtyBitmap::from_phys_chunk_epochs(&chunk_epochs, 10, 10, row_len, elem_size);
        assert_eq!(bitmap.total_pages(), 4);
        assert_eq!(bitmap.dirty_page_count(), 2);
        assert!(bitmap.is_page_dirty(0));
        assert!(!bitmap.is_page_dirty(1));
        assert!(!bitmap.is_page_dirty(2));
        assert!(bitmap.is_page_dirty(3));

        let runs = bitmap.dirty_runs();
        assert_eq!(runs.len(), 2);
        assert_eq!(
            runs[0],
            DirtyPageRun {
                start_page: 0,
                page_count: 1,
                byte_offset: 0,
                byte_len: HARDWARE_PAGE_SIZE,
            }
        );
        assert_eq!(
            runs[1],
            DirtyPageRun {
                start_page: 3,
                page_count: 1,
                byte_offset: 3 * HARDWARE_PAGE_SIZE,
                byte_len: HARDWARE_PAGE_SIZE,
            }
        );
    }

    #[test]
    fn test_cow_gpu_column_pin_zero_blocking_freeze_and_tail_zero_pad() {
        let mut store = Store::new();
        for i in 0..100u128 {
            store
                .insert((
                    EntityId::from_u128(1000 + i),
                    (Pos3D {
                        x: i as f64,
                        y: 1.0,
                        z: 2.0,
                        w: 1.0,
                    },),
                ))
                .unwrap();
        }

        let arch_id = store
            .archetypes
            .iter()
            .find(|a| a.len() == 100)
            .unwrap()
            .id()
            .0;

        let pin = store.pin_column_for_gpu::<Pos3D>(arch_id).unwrap();
        assert_eq!(pin.len(), 100);
        assert_eq!(pin.as_slice()[0].x, 0.0);

        store
            .modify::<Pos3D, _>(EntityId::from_u128(1000), |p| p.x = 9999.0)
            .unwrap();

        assert_eq!(
            pin.as_slice()[0].x,
            0.0,
            "Pinned GPU column view must remain immutable across CPU COW writes"
        );
        assert_eq!(
            store.get::<Pos3D>(EntityId::from_u128(1000)).unwrap().x,
            9999.0
        );

        let uma_buf = pin.to_page_aligned_buffer();
        let desc = MetalNoCopyDescriptor::from_page_aligned_buffer(&uma_buf);
        assert!(desc.is_page_aligned());
        assert_eq!(desc.allocated_bytes, HARDWARE_PAGE_SIZE);
        assert_eq!(desc.used_bytes, 100 * 32);

        let tail_slice = unsafe {
            std::slice::from_raw_parts(
                desc.base_ptr.add(desc.used_bytes),
                desc.allocated_bytes - desc.used_bytes,
            )
        };
        assert!(tail_slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_gpu_compute_pass_with_row_flags_and_changed_since_sync() {
        let mut store = Store::new();
        for i in 0..600u128 {
            let e = EntityId::from_u128(5000 + i);
            store
                .insert((
                    e,
                    (
                        Pos3D {
                            x: i as f64,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        Vel3D {
                            vx: 10.0,
                            vy: 5.0,
                            vz: 0.0,
                            vw: 0.0,
                        },
                    ),
                ))
                .unwrap();
            if i == 10 || i == 20 {
                store.set_flag::<0>(e, true).unwrap();
            }
        }

        let epoch_before_gpu = store.epoch();

        let pass = GpuComputePass::new().with_flag::<0>();
        let receipt = pass
            .dispatch_binary_in_place::<Pos3D, Vel3D, _>(&mut store, |_row, pos, vel| {
                pos.x += vel.vx;
                pos.y += vel.vy;
            })
            .unwrap();

        assert_eq!(receipt.total_rows_scanned, 600);
        assert_eq!(receipt.total_active_rows, 2);
        assert_eq!(receipt.total_skipped_64row_chunks, 9);
        assert_eq!(receipt.total_dirty_bytes, HARDWARE_PAGE_SIZE);
        assert_eq!(receipt.total_column_page_bytes, 2 * HARDWARE_PAGE_SIZE);
        assert!((receipt.dma_bandwidth_saved_ratio() - 0.5).abs() < 1e-6);

        let q_changed = QueryBuilder::<&Pos3D>::new()
            .changed_since::<Pos3D>(epoch_before_gpu)
            .build(&store)
            .unwrap();
        let mut changed_count = 0usize;
        q_changed
            .for_each(&store, |(_id, _pos)| changed_count += 1)
            .unwrap();
        assert_eq!(changed_count, 64);
    }

    #[test]
    fn test_cross_row_relational_pair_gather_and_scatter_reduce() {
        let mut store = Store::new();
        let anchor_a = EntityId::from_u128(1);
        let anchor_b = EntityId::from_u128(2);
        store
            .insert((
                anchor_a,
                (
                    Pos3D {
                        x: 100.0,
                        y: 200.0,
                        z: 0.0,
                        w: 1.0,
                    },
                    ForceAccum { fx: 0.0, fy: 0.0 },
                ),
            ))
            .unwrap();
        store
            .insert((
                anchor_b,
                (
                    Pos3D {
                        x: -50.0,
                        y: -100.0,
                        z: 0.0,
                        w: 1.0,
                    },
                    ForceAccum { fx: 0.0, fy: 0.0 },
                ),
            ))
            .unwrap();

        for i in 0..20u128 {
            let target = if i % 2 == 0 { anchor_a } else { anchor_b };
            store
                .insert((
                    EntityId::from_u128(100 + i),
                    (
                        Pos3D {
                            x: 1.0,
                            y: 2.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        ForceAccum { fx: 3.0, fy: 4.0 },
                        crate::al::Pair::new(AttachedTo, target),
                    ),
                ))
                .unwrap();
        }

        let pass = GpuComputePass::new();
        let gather_receipt = pass
            .dispatch_pair_gather::<AttachedTo, Pos3D, Pos3D, _>(
                &mut store,
                |_row, child_pos, anchor_pos| {
                    child_pos.x += anchor_pos.x;
                    child_pos.y += anchor_pos.y;
                },
            )
            .unwrap();
        assert_eq!(gather_receipt.total_active_rows, 20);
        assert_eq!(
            store.get::<Pos3D>(EntityId::from_u128(100)).unwrap().x,
            101.0
        );
        assert_eq!(
            store.get::<Pos3D>(EntityId::from_u128(101)).unwrap().x,
            -49.0
        );

        let scatter_receipt = pass
            .dispatch_pair_scatter_reduce::<AttachedTo, ForceAccum, ForceAccum, _>(
                &mut store,
                |anchor_force, child_force| {
                    anchor_force.fx += child_force.fx;
                    anchor_force.fy += child_force.fy;
                },
            )
            .unwrap();
        assert_eq!(scatter_receipt.total_active_rows, 20);
        assert_eq!(
            store.get::<ForceAccum>(anchor_a).unwrap(),
            &ForceAccum { fx: 30.0, fy: 40.0 }
        );
        assert_eq!(
            store.get::<ForceAccum>(anchor_b).unwrap(),
            &ForceAccum { fx: 30.0, fy: 40.0 }
        );

        let (total_fx, _) = pass.dispatch_reduce_f64::<ForceAccum, _>(&store, |f| f.fx);
        assert!((total_fx - (30.0 + 30.0 + 20.0 * 3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_kernel_ir_fp64_gather_and_multi_backend_codegen() {
        let mut ir = KernelIr::new("nbody_relational_step", 64);
        let _mask_slot = ir.enable_bitmask_predication();
        let pos_slot = ir.add_binding("pos", GpuScalarKind::F64, 1, BufferAccess::ReadWrite);
        let target_pos_slot =
            ir.add_binding("target_pos", GpuScalarKind::F64, 1, BufferAccess::ReadOnly);
        let lut_slot = ir.add_binding(
            "pair_lut",
            GpuScalarKind::U32,
            1,
            BufferAccess::PairTargetLut,
        );
        let energy_slot = ir.add_binding(
            "total_energy",
            GpuScalarKind::F64,
            1,
            BufferAccess::AtomicReadWrite,
        );

        ir.push_stmt(IrStmt::Let {
            name: "t_row".into(),
            scalar: GpuScalarKind::U32,
            value: IrExpr::PairTargetRow {
                lut_binding: lut_slot,
            },
        });

        ir.push_stmt(IrStmt::Let {
            name: "anchor_x".into(),
            scalar: GpuScalarKind::F64,
            value: IrExpr::GatherLoad {
                binding: target_pos_slot,
                target_row: Box::new(IrExpr::Var("t_row".into())),
                elem_offset: 0,
                scalar: GpuScalarKind::F64,
                fallback: Box::new(IrExpr::f64_const(0.0)),
            },
        });

        ir.push_stmt(IrStmt::Let {
            name: "new_x".into(),
            scalar: GpuScalarKind::F64,
            value: IrExpr::Var("anchor_x".into()).fma(
                IrExpr::f64_const(0.5),
                IrExpr::RowLoad {
                    binding: pos_slot,
                    elem_offset: 0,
                    scalar: GpuScalarKind::F64,
                },
            ),
        });

        ir.push_stmt(IrStmt::RowStore {
            binding: pos_slot,
            elem_offset: 0,
            scalar: GpuScalarKind::F64,
            value: IrExpr::Var("new_x".into()),
        });

        ir.push_stmt(IrStmt::ThreadgroupReduceSum {
            dst_binding: energy_slot,
            scalar: GpuScalarKind::F64,
            value: IrExpr::Var("new_x".into()),
        });

        assert!(ir.uses_fp64());
        assert!(ir.uses_cross_row_gather());

        let wgsl = ir.compile_wgsl();
        assert!(wgsl.contains("@compute @workgroup_size(64, 1, 1)"));
        assert!(wgsl.contains("__row_mask"));
        assert!(
            wgsl.contains("select(0.000000lf, target_pos[t_row], (t_row < __params.target_row_count))")
        );

        let msl = ir.compile_msl();
        assert!(msl.contains("kernel void nbody_relational_step"));
        assert!(msl.contains("device double* pos"));
        assert!(msl.contains("simd_sum(new_x)"));

        let cuda = ir.compile_cuda();
        assert!(cuda.contains("extern \"C\" __global__ void nbody_relational_step"));
        assert!(cuda.contains("const double* __restrict__ target_pos"));
        assert!(cuda.contains("atomicAdd(&total_energy[0], new_x)"));
    }
}
