//! 多平台 GPU Compute 内核抽象、着色器生成 (MSL / WGSL) 与三层并行算子体系 (`#[gpu]`)
//!
//! 支持在同一份 Rust 代码中定义组件与内核函数：
//! ```rust,ignore
//! #[gpu]
//! pub struct SimPosition { pub x: f32, pub y: f32 }
//!
//! #[gpu]
//! pub struct SimVelocity { pub vx: f32, pub vy: f32 }
//!
//! #[gpu]
//! pub fn integrate_sim_motion(pos: &mut SimPosition, vel: &SimVelocity) {
//!     pos.x += vel.vx;
//!     pos.y += vel.vy;
//! }
//! ```
//! 并支持三层显式遍历方式：
//! 1. **CPU 单线程**：闭包 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.for_each(|(pos, vel)| { pos.x += vel.vx; })`
//!    或算子直传 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.run(integrate_sim_motion)`
//! 2. **CPU 多核并行 (Rayon)**：闭包 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.par_for_each(|(pos, vel)| ...)`
//!    或算子直传 `.par_run(integrate_sim_motion)` / `.par().run(integrate_sim_motion)`
//! 3. **GPU 硬件并行 (16KB 页对齐零拷贝)**：`store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.gpu().for_each(integrate_sim_motion)?`
//!    或 `.gpu().run(integrate_sim_motion)?`

use super::GpuComponent;
use crate::id::ComponentTypeId;
use crate::storage::query::QueryMutData;
pub use crate::storage::query::{
    ClosureOpMarker, KernelOpMarker, QueryForEachOp, QueryParForEachOp, QueryRowItem,
};
use bytemuck::{Pod, Zeroable};

/// 传递给 GPU Compute Shader 常量缓冲区的派发元信息 (`[[buffer(N)]]` / `@group(0) @binding(N)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDispatchUniforms {
    /// 当前 Archetype 的有效元素行数（线程越界保护上界 `if (gid >= uniforms.active_rows) return;`）
    pub active_rows: u32,
    /// 每个 Threadgroup / Workgroup 的线程数
    pub threads_per_group: u32,
    /// 当前派发的 Archetype ID
    pub archetype_id: u32,
    /// 16 字节对齐保留位
    pub _pad: u32,
}

/// 跨平台着色器字段 32-bit 标量/紧凑数组类型枚举（保证 Rust `#[repr(C)]`、Apple MSL 与 WebGPU WGSL 内存布局 100% 一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFieldType {
    F32,
    I32,
    U32,
    Vec2F32,
    Vec3F32,
    Vec4F32,
    Vec2U32,
    Vec3U32,
    Vec4U32,
    Vec2I32,
    Vec3I32,
    Vec4I32,
}

impl GpuFieldType {
    /// 映射为 Apple Metal Shading Language (MSL) 类型名（4 字节对齐）。
    #[inline]
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::Vec2F32 => "packed_float2",
            Self::Vec3F32 => "packed_float3",
            Self::Vec4F32 => "packed_float4",
            Self::Vec2U32 => "packed_uint2",
            Self::Vec3U32 => "packed_uint3",
            Self::Vec4U32 => "packed_uint4",
            Self::Vec2I32 => "packed_int2",
            Self::Vec3I32 => "packed_int3",
            Self::Vec4I32 => "packed_int4",
        }
    }

    /// 映射为 WebGPU Shading Language (WGSL) 类型名（4 字节对齐，与 Rust 数组及 MSL `packed_*` 布局完全一致）。
    #[inline]
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Vec2F32 => "array<f32, 2>",
            Self::Vec3F32 => "array<f32, 3>",
            Self::Vec4F32 => "array<f32, 4>",
            Self::Vec2U32 => "array<u32, 2>",
            Self::Vec3U32 => "array<u32, 3>",
            Self::Vec4U32 => "array<u32, 4>",
            Self::Vec2I32 => "array<i32, 2>",
            Self::Vec3I32 => "array<i32, 3>",
            Self::Vec4I32 => "array<i32, 4>",
        }
    }

    /// 字段字节大小。
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Vec2F32 | Self::Vec2U32 | Self::Vec2I32 => 8,
            Self::Vec3F32 | Self::Vec3U32 | Self::Vec3I32 => 12,
            Self::Vec4F32 | Self::Vec4U32 | Self::Vec4I32 => 16,
        }
    }

    /// 字段字节对齐要求（所有跨平台着色器标量/紧凑数组统一为 4 字节对齐）。
    #[inline]
    pub const fn byte_align(self) -> usize {
        4
    }
}

/// Rust 基础 32-bit 数值/数组类型到 `GpuFieldType` 的编译期映射 Trait。
pub trait GpuScalarType: Copy + Send + Sync + 'static {
    const FIELD_TYPE: GpuFieldType;
}

impl GpuScalarType for f32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::F32;
}
impl GpuScalarType for i32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::I32;
}
impl GpuScalarType for u32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::U32;
}
impl GpuScalarType for [f32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2F32;
}
impl GpuScalarType for [f32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3F32;
}
impl GpuScalarType for [f32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4F32;
}
impl GpuScalarType for [u32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2U32;
}
impl GpuScalarType for [u32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3U32;
}
impl GpuScalarType for [u32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4U32;
}
impl GpuScalarType for [i32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2I32;
}
impl GpuScalarType for [i32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3I32;
}
impl GpuScalarType for [i32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4I32;
}

/// GPU 组件结构体字段元数据。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuStructField {
    pub name: &'static str,
    pub field_type: GpuFieldType,
}

/// 具备自动生成 MSL / WGSL 结构体定义能力的 GPU 组件契约。
pub trait GpuShaderStruct: GpuComponent {
    /// 着色器内的结构体名称。
    const STRUCT_NAME: &'static str;
    /// 字段列表。
    const FIELDS: &'static [GpuStructField];

    /// 校验 Rust `#[repr(C)]` 结构体布局与着色器字段无隐式填充字节差异。
    #[inline]
    fn assert_shader_layout() {
        <Self as GpuComponent>::assert_gpu_layout();
        let expected_size: usize = Self::FIELDS.iter().map(|f| f.field_type.byte_size()).sum();
        assert_eq!(
            std::mem::size_of::<Self>(),
            expected_size,
            "GpuShaderStruct `{}` has unrepresented padding bytes (Rust size {} != shader field sum {})",
            Self::STRUCT_NAME,
            std::mem::size_of::<Self>(),
            expected_size
        );
    }

    /// 生成 Apple Metal Shading Language (MSL) `struct` 声明。
    fn msl_struct_definition() -> String {
        let mut s = format!("struct {} {{\n", Self::STRUCT_NAME);
        for f in Self::FIELDS {
            s.push_str(&format!("    {} {};\n", f.field_type.msl_type(), f.name));
        }
        s.push_str("};\n");
        s
    }

    /// 生成 WebGPU Shading Language (WGSL) `struct` 声明。
    fn wgsl_struct_definition() -> String {
        let mut s = format!("struct {} {{\n", Self::STRUCT_NAME);
        for f in Self::FIELDS {
            s.push_str(&format!("    {}: {},\n", f.name, f.field_type.wgsl_type()));
        }
        s.push_str("};\n");
        s
    }
}

/// 内核单个列参数的绑定规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuKernelBindingSpec {
    /// 内核形参名（如 `"pos"`、`"vel"`）
    pub param_name: &'static str,
    /// 组件结构体名（如 `"SimPosition"`）
    pub struct_name: &'static str,
    /// 组件 `ComponentTypeId`
    pub type_id: ComponentTypeId,
    /// 单个元素字节大小
    pub elem_size: usize,
    /// 是否为可变写绑定 (`&mut T`)
    pub mutable: bool,
}

/// CPU 镜像/回退整列块执行函数签名。
pub type GpuCpuChunkRunner = unsafe fn(
    raw_ptrs: &[*mut u8],
    active_rows: usize,
    uniforms: &GpuDispatchUniforms,
);

/// 完整的跨平台 GPU Compute 内核描述符（包含 MSL 源码、WGSL 源码、列绑定规格与 CPU 镜像执行器）。
///
/// 字段对外只读，仅能通过 `unsafe fn new`（或 `#[gpu]` 宏）构造，防止外部伪造错配的 `bindings` 与 `cpu_runner`。
#[derive(Debug, Clone)]
pub struct GpuKernelDescriptor {
    /// 内核函数名称
    name: &'static str,
    /// 列参数绑定表（顺序与 `Query` 元组严格一致）
    bindings: Vec<GpuKernelBindingSpec>,
    /// 自动生成的 Apple Metal Shading Language (MSL) 完整内核源码
    msl_source: String,
    /// 自动生成的 WebGPU Shading Language (WGSL) 完整内核源码
    wgsl_source: String,
    /// CPU 回退 / UMA 镜像执行函数
    cpu_runner: GpuCpuChunkRunner,
}

impl GpuKernelDescriptor {
    /// 构造内核描述符。
    ///
    /// # Safety
    /// 调用方必须保证 `bindings` 与 `cpu_runner` 解引用的列指针数量、类型大小、对齐及读写权限严格一致。
    #[inline]
    pub unsafe fn new(
        name: &'static str,
        bindings: Vec<GpuKernelBindingSpec>,
        msl_source: String,
        wgsl_source: String,
        cpu_runner: GpuCpuChunkRunner,
    ) -> Self {
        Self {
            name,
            bindings,
            msl_source,
            wgsl_source,
            cpu_runner,
        }
    }

    /// 内核函数名称。
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 列参数绑定规格列表。
    #[inline]
    pub fn bindings(&self) -> &[GpuKernelBindingSpec] {
        &self.bindings
    }

    /// Apple Metal Shading Language (MSL) 完整源码。
    #[inline]
    pub fn msl_source(&self) -> &str {
        &self.msl_source
    }

    /// WebGPU Shading Language (WGSL) 完整源码。
    #[inline]
    pub fn wgsl_source(&self) -> &str {
        &self.wgsl_source
    }

    /// CPU 整列块回退执行器。
    #[inline]
    pub fn cpu_runner(&self) -> GpuCpuChunkRunner {
        self.cpu_runner
    }
}

/// 强类型 GPU 计算内核契约：将内核结构体静态绑定至对应的 ECS 可变查询类型 `Self::Query`。
///
/// # Safety
/// 实现者必须保证 `descriptor()` 返回的 `bindings` 与 `Self::Query` 的列顺序、类型、`elem_size` 及可变性完全一致，
/// 且 `cpu_runner` 仅按 `bindings` 声明的权限与长度访问列切片。
pub unsafe trait GpuKernel<'a>: Copy + Send + Sync + 'static {
    /// 匹配的 `Store::query_mut::<Q>()` 查询类型参数（如 `(&'a mut SimPosition, &'a SimVelocity)`）。
    type Query: QueryMutData<'a>;

    /// 获取静态缓存的多平台编译描述符（MSL + WGSL + CPU Runner）。
    fn descriptor(&self) -> &'static GpuKernelDescriptor;
}

/// 根据结构体定义、绑定表与已翻译的 MSL 函数体生成完整的 Apple Metal Shading Language (MSL) 源码。
pub fn build_msl_kernel_source(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    msl_body: &str,
) -> String {
    build_msl_kernel_source_with_helpers(kernel_name, struct_defs, bindings, "", msl_body)
}

/// 根据结构体定义、绑定表、前置 `#[gpu]` 辅助函数与已翻译的 MSL 函数体生成完整的 MSL 源码。
pub fn build_msl_kernel_source_with_helpers(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    msl_helpers: &str,
    msl_body: &str,
) -> String {
    let mut src = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         struct GpuDispatchUniforms {\n\
         \x20   uint active_rows;\n\
         \x20   uint threads_per_group;\n\
         \x20   uint archetype_id;\n\
         \x20   uint _pad;\n\
         };\n\n",
    );
    let mut seen_structs: Vec<(&str, ComponentTypeId)> = Vec::new();
    for (idx, b) in bindings.iter().enumerate() {
        if let Some((_, existing_tid)) = seen_structs.iter().find(|(name, _)| *name == b.struct_name)
        {
            assert_eq!(
                *existing_tid, b.type_id,
                "GPU kernel `{kernel_name}` binds two distinct component types with the same shader struct name `{}`",
                b.struct_name
            );
            continue;
        }
        seen_structs.push((b.struct_name, b.type_id));
        if let Some(def) = struct_defs.get(idx) {
            src.push_str(def);
            src.push('\n');
        }
    }
    if !msl_helpers.trim().is_empty() {
        src.push_str(msl_helpers.trim());
        src.push_str("\n\n");
    }

    src.push_str(&format!("kernel void {kernel_name}(\n"));
    for (idx, b) in bindings.iter().enumerate() {
        let qualifier = if b.mutable { "device" } else { "const device" };
        src.push_str(&format!(
            "    {qualifier} {}* sumeru_buf_{} [[buffer({idx})]],\n",
            b.struct_name, b.param_name
        ));
    }
    let uniform_idx = bindings.len();
    src.push_str(&format!(
        "    constant GpuDispatchUniforms& uniforms [[buffer({uniform_idx})]],\n\
         \x20   uint gid [[thread_position_in_grid]]\n\
         ) {{\n\
         \x20   if (gid >= uniforms.active_rows) {{ return; }}\n"
    ));
    for b in bindings {
        let qualifier = if b.mutable { "device" } else { "const device" };
        src.push_str(&format!(
            "    {qualifier} {}& {} = sumeru_buf_{}[gid];\n",
            b.struct_name, b.param_name, b.param_name
        ));
    }
    src.push_str("    ");
    src.push_str(msl_body.trim());
    src.push_str("\n}\n");
    src
}

/// 根据结构体定义、绑定表与已翻译的 WGSL 函数体生成完整的 WebGPU Shading Language (WGSL) 源码。
pub fn build_wgsl_kernel_source(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    wgsl_body: &str,
) -> String {
    build_wgsl_kernel_source_with_helpers(kernel_name, struct_defs, bindings, "", wgsl_body)
}

/// 根据结构体定义、绑定表、前置 `#[gpu]` 辅助函数与已翻译的 WGSL 函数体生成完整的 WGSL 源码。
pub fn build_wgsl_kernel_source_with_helpers(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    wgsl_helpers: &str,
    wgsl_body: &str,
) -> String {
    let mut src = String::from(
        "struct GpuDispatchUniforms {\n\
         \x20   active_rows: u32,\n\
         \x20   threads_per_group: u32,\n\
         \x20   archetype_id: u32,\n\
         \x20   _pad: u32,\n\
         };\n\n",
    );
    let mut seen_structs: Vec<(&str, ComponentTypeId)> = Vec::new();
    for (idx, b) in bindings.iter().enumerate() {
        if let Some((_, existing_tid)) = seen_structs.iter().find(|(name, _)| *name == b.struct_name)
        {
            assert_eq!(
                *existing_tid, b.type_id,
                "GPU kernel `{kernel_name}` binds two distinct component types with the same shader struct name `{}`",
                b.struct_name
            );
            continue;
        }
        seen_structs.push((b.struct_name, b.type_id));
        if let Some(def) = struct_defs.get(idx) {
            src.push_str(def);
            src.push('\n');
        }
    }
    if !wgsl_helpers.trim().is_empty() {
        src.push_str(wgsl_helpers.trim());
        src.push_str("\n\n");
    }
    for (idx, b) in bindings.iter().enumerate() {
        let access = if b.mutable { "read_write" } else { "read" };
        src.push_str(&format!(
            "@group(0) @binding({idx}) var<storage, {access}> sumeru_buf_{}: array<{}>;\n",
            b.param_name, b.struct_name
        ));
    }
    let uniform_idx = bindings.len();
    src.push_str(&format!(
        "@group(0) @binding({uniform_idx}) var<uniform> uniforms: GpuDispatchUniforms;\n\n\
         @compute @workgroup_size(256)\n\
         fn {kernel_name}(@builtin(global_invocation_id) global_id: vec3<u32>) {{\n\
         \x20   let gid = global_id.x;\n\
         \x20   if (gid >= uniforms.active_rows) {{ return; }}\n"
    ));
    for b in bindings {
        let kw = if b.mutable { "var" } else { "let" };
        src.push_str(&format!(
            "    {kw} {} = sumeru_buf_{}[gid];\n",
            b.param_name, b.param_name
        ));
    }
    src.push_str("    ");
    src.push_str(wgsl_body.trim());
    src.push('\n');
    for b in bindings {
        if b.mutable {
            src.push_str(&format!(
                "    sumeru_buf_{}[gid] = {};\n",
                b.param_name, b.param_name
            ));
        }
    }
    src.push_str("}\n");
    src
}

/// 声明式 GPU 组件定义宏（由 `#[gpu]` 与 `#[gpu_component]` 展开调用，也可直接使用）。
#[macro_export]
macro_rules! sumeru_gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $Name:ident {
            $(
                $(#[$fmeta:meta])*
                $fvis:vis $fname:ident : $fty:ty
            ),* $(,)?
        }
    ) => {
        #[repr(C)]
        $(#[$meta])*
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            PartialEq,
            $crate::__private::bytemuck::Pod,
            $crate::__private::bytemuck::Zeroable,
            $crate::__private::serde::Serialize,
            $crate::__private::serde::Deserialize,
            $crate::__private::rkyv::Archive,
            $crate::__private::rkyv::Serialize,
            $crate::__private::rkyv::Deserialize,
        )]
        $vis struct $Name {
            $(
                $(#[$fmeta])*
                $fvis $fname : $fty,
            )*
        }

        // SAFETY: #[repr(C)] + bytemuck::Pod + Zeroable guaranteed above.
        unsafe impl $crate::GpuComponent for $Name {}

        impl $crate::GpuShaderStruct for $Name {
            const STRUCT_NAME: &'static str = stringify!($Name);
            const FIELDS: &'static [$crate::GpuStructField] = &[
                $(
                    $crate::GpuStructField {
                        name: stringify!($fname),
                        field_type: <$fty as $crate::GpuScalarType>::FIELD_TYPE,
                    },
                )*
            ];
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_param_type {
    ($lt:lifetime, [mut] $T:ty) => {
        &$lt mut $T
    };
    ($lt:lifetime, [ref] $T:ty) => {
        &$lt $T
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_is_mut {
    ([mut]) => {
        true
    };
    ([ref]) => {
        false
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_slice_from_raw {
    ([mut] $T:ty, $ptr:expr, $len:expr) => {
        ::std::slice::from_raw_parts_mut($ptr as *mut $T, $len)
    };
    ([ref] $T:ty, $ptr:expr, $len:expr) => {
        ::std::slice::from_raw_parts($ptr as *const $T, $len)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_elem_borrow {
    ([mut] $slice:ident, $idx:ident) => {
        &mut $slice[$idx]
    };
    ([ref] $slice:ident, $idx:ident) => {
        &$slice[$idx]
    };
}

/// 统一 `#[gpu]` 函数展开宏：生成同名可调用算子结构体与静态描述符，同时支持：
/// - 在普通代码或闭包中直接调用：`integrate_sim_motion(pos, vel)`
/// - 直接传入 CPU 单线程遍历：`.run(integrate_sim_motion)`
/// - 直接传入 CPU 多核并行遍历：`.par_run(integrate_sim_motion)` / `.par().run(integrate_sim_motion)`
/// - 直接传入 GPU 硬件并行遍历：`.gpu().for_each(integrate_sim_motion)?` / `.gpu().run(integrate_sim_motion)?`
#[macro_export]
macro_rules! sumeru_gpu_fn {
    // 1 parameter (with pre-translated MSL & WGSL bodies)
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call($p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0)) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                for idx in 0..active_rows {
                    Self::call($crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx));
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a> fn($crate::__sumeru_gpu_param_type!('a, [$m0] $T0));

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a> fn($crate::__sumeru_gpu_param_type!('a, [$m0] $T0)) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0)), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(&mut self, (_entity, item): ($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0))) {
                Self::call(item);
            }
        }

        impl<'a> $crate::QueryParForEachOp<($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0)), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(&self, (_entity, item): ($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0))) {
                Self::call(item);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = $crate::__sumeru_gpu_param_type!('a, [$m0] $T0);

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                    ];
                    let msl_defs = vec![<$T0 as $crate::GpuShaderStruct>::msl_struct_definition()];
                    let wgsl_defs = vec![<$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition()];
                    let msl_source = $crate::build_msl_kernel_source(stringify!($fn_name), &msl_defs, &bindings, $msl_body);
                    let wgsl_source = $crate::build_wgsl_kernel_source(stringify!($fn_name), &wgsl_defs, &bindings, $wgsl_body);
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 2 parameters (with pre-translated MSL & WGSL bodies)
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                    ),
                ),
            ) {
                Self::call(a, b);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                    ),
                ),
            ) {
                Self::call(a, b);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let msl_source = $crate::build_msl_kernel_source(stringify!($fn_name), &msl_defs, &bindings, $msl_body);
                    let wgsl_source = $crate::build_wgsl_kernel_source(stringify!($fn_name), &wgsl_defs, &bindings, $wgsl_body);
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 3 parameters (with pre-translated MSL & WGSL bodies)
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty,
            $p2:ident : [$m2:ident] $T2:ty
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
                $p2: $crate::__sumeru_gpu_param_type!('_, [$m2] $T2),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                let s2 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m2] $T2, raw_ptrs[2], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m2] s2, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b, 'c> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b, 'c> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                    $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b, c)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                    ),
                ),
            ) {
                Self::call(a, b, c);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b, c)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                    ),
                ),
            ) {
                Self::call(a, b, c);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T2 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T2>(),
                            elem_size: ::std::mem::size_of::<$T2>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let msl_source = $crate::build_msl_kernel_source(stringify!($fn_name), &msl_defs, &bindings, $msl_body);
                    let wgsl_source = $crate::build_wgsl_kernel_source(stringify!($fn_name), &wgsl_defs, &bindings, $wgsl_body);
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 4 parameters (with pre-translated MSL & WGSL bodies)
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty,
            $p2:ident : [$m2:ident] $T2:ty,
            $p3:ident : [$m3:ident] $T3:ty
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
                $p2: $crate::__sumeru_gpu_param_type!('_, [$m2] $T2),
                $p3: $crate::__sumeru_gpu_param_type!('_, [$m3] $T3),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                let s2 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m2] $T2, raw_ptrs[2], active_rows) };
                let s3 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m3] $T3, raw_ptrs[3], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m2] s2, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m3] s3, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b, 'c, 'd> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('d, [$m3] $T3),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b, 'c, 'd> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                    $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                    $crate::__sumeru_gpu_param_type!('d, [$m3] $T3),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b, c, d)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                        $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
                    ),
                ),
            ) {
                Self::call(a, b, c, d);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b, c, d)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                        $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
                    ),
                ),
            ) {
                Self::call(a, b, c, d);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T2 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T3 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T2>(),
                            elem_size: ::std::mem::size_of::<$T2>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p3),
                            struct_name: <$T3 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T3>(),
                            elem_size: ::std::mem::size_of::<$T3>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m3]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T3 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T3 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let msl_source = $crate::build_msl_kernel_source(stringify!($fn_name), &msl_defs, &bindings, $msl_body);
                    let wgsl_source = $crate::build_wgsl_kernel_source(stringify!($fn_name), &wgsl_defs, &bindings, $wgsl_body);
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 1 parameter (Typed Shader IR)
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call($p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0)) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                for idx in 0..active_rows {
                    Self::call($crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx));
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a> fn($crate::__sumeru_gpu_param_type!('a, [$m0] $T0));

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a> fn($crate::__sumeru_gpu_param_type!('a, [$m0] $T0)) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0)), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(&mut self, (_entity, item): ($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0))) {
                Self::call(item);
            }
        }

        impl<'a> $crate::QueryParForEachOp<($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0)), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(&self, (_entity, item): ($crate::EntityId, $crate::__sumeru_gpu_param_type!('a, [$m0] $T0))) {
                Self::call(item);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = $crate::__sumeru_gpu_param_type!('a, [$m0] $T0);

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                    ];
                    let components = [
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T0 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                    ];
                    let msl_defs = vec![<$T0 as $crate::GpuShaderStruct>::msl_struct_definition()];
                    let wgsl_defs = vec![<$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition()];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources(
                        stringify!($fn_name),
                        &components,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 2 parameters (Typed Shader IR)
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                    ),
                ),
            ) {
                Self::call(a, b);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                    ),
                ),
            ) {
                Self::call(a, b);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                    ];
                    let components = [
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T0 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T1 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources(
                        stringify!($fn_name),
                        &components,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 3 parameters (Typed Shader IR)
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty,
            $p2:ident : [$m2:ident] $T2:ty
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
                $p2: $crate::__sumeru_gpu_param_type!('_, [$m2] $T2),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                let s2 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m2] $T2, raw_ptrs[2], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m2] s2, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b, 'c> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b, 'c> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                    $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b, c)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                    ),
                ),
            ) {
                Self::call(a, b, c);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b, c)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                    ),
                ),
            ) {
                Self::call(a, b, c);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T2 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T2>(),
                            elem_size: ::std::mem::size_of::<$T2>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                    ];
                    let components = [
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T0 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T1 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T2 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources(
                        stringify!($fn_name),
                        &components,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // 4 parameters (Typed Shader IR)
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $p0:ident : [$m0:ident] $T0:ty,
            $p1:ident : [$m1:ident] $T1:ty,
            $p2:ident : [$m2:ident] $T2:ty,
            $p3:ident : [$m3:ident] $T3:ty
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            pub fn call(
                $p0: $crate::__sumeru_gpu_param_type!('_, [$m0] $T0),
                $p1: $crate::__sumeru_gpu_param_type!('_, [$m1] $T1),
                $p2: $crate::__sumeru_gpu_param_type!('_, [$m2] $T2),
                $p3: $crate::__sumeru_gpu_param_type!('_, [$m3] $T3),
            ) {
                $($body)*
            }

            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let s0 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m0] $T0, raw_ptrs[0], active_rows) };
                let s1 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m1] $T1, raw_ptrs[1], active_rows) };
                let s2 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m2] $T2, raw_ptrs[2], active_rows) };
                let s3 = unsafe { $crate::__sumeru_gpu_slice_from_raw!([$m3] $T3, raw_ptrs[3], active_rows) };
                for idx in 0..active_rows {
                    Self::call(
                        $crate::__sumeru_gpu_elem_borrow!([$m0] s0, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m1] s1, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m2] s2, idx),
                        $crate::__sumeru_gpu_elem_borrow!([$m3] s3, idx),
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = for<'a, 'b, 'c, 'd> fn(
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('d, [$m3] $T3),
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: for<'a, 'b, 'c, 'd> fn(
                    $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                    $crate::__sumeru_gpu_param_type!('b, [$m1] $T1),
                    $crate::__sumeru_gpu_param_type!('c, [$m2] $T2),
                    $crate::__sumeru_gpu_param_type!('d, [$m3] $T3),
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row(
                &mut self,
                (_entity, (a, b, c, d)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                        $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
                    ),
                ),
            ) {
                Self::call(a, b, c, d);
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            ),
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            fn call_row_par(
                &self,
                (_entity, (a, b, c, d)): (
                    $crate::EntityId,
                    (
                        $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                        $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                        $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                        $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
                    ),
                ),
            ) {
                Self::call(a, b, c, d);
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = (
                $crate::__sumeru_gpu_param_type!('a, [$m0] $T0),
                $crate::__sumeru_gpu_param_type!('a, [$m1] $T1),
                $crate::__sumeru_gpu_param_type!('a, [$m2] $T2),
                $crate::__sumeru_gpu_param_type!('a, [$m3] $T3),
            );

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    <$T0 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T1 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T2 as $crate::GpuShaderStruct>::assert_shader_layout();
                    <$T3 as $crate::GpuShaderStruct>::assert_shader_layout();
                    let bindings = vec![
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T0>(),
                            elem_size: ::std::mem::size_of::<$T0>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T1>(),
                            elem_size: ::std::mem::size_of::<$T1>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T2>(),
                            elem_size: ::std::mem::size_of::<$T2>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                        $crate::GpuKernelBindingSpec {
                            param_name: stringify!($p3),
                            struct_name: <$T3 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            type_id: $crate::ComponentTypeId::of::<$T3>(),
                            elem_size: ::std::mem::size_of::<$T3>(),
                            mutable: $crate::__sumeru_gpu_is_mut!([$m3]),
                        },
                    ];
                    let components = [
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p0),
                            struct_name: <$T0 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T0 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m0]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p1),
                            struct_name: <$T1 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T1 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m1]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p2),
                            struct_name: <$T2 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T2 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m2]),
                        },
                        $crate::GpuComponentSchema {
                            param_name: stringify!($p3),
                            struct_name: <$T3 as $crate::GpuShaderStruct>::STRUCT_NAME,
                            fields: <$T3 as $crate::GpuShaderStruct>::FIELDS,
                            mutable: $crate::__sumeru_gpu_is_mut!([$m3]),
                        },
                    ];
                    let msl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::msl_struct_definition(),
                        <$T3 as $crate::GpuShaderStruct>::msl_struct_definition(),
                    ];
                    let wgsl_defs = vec![
                        <$T0 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T1 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T2 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                        <$T3 as $crate::GpuShaderStruct>::wgsl_struct_definition(),
                    ];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources(
                        stringify!($fn_name),
                        &components,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // Fallback @normalized arms for direct macro_rules! callers
    (
        @normalized
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident ( $($params:tt)* ) { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @translated
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            msl = stringify!($($body)*),
            wgsl = stringify!($($body)*),
            body = { $($body)* }
        }
    };
}

/// 显式内核类型名展开宏（供 `#[gpu_kernel(CustomKernelName)]` 使用）。
#[macro_export]
macro_rules! sumeru_gpu_kernel {
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis kernel $KernelType:ident fn $fn_name:ident (
            $($params:tt)*
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @ir
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            ir = $ir_block,
            body = { $($body)* }
        }
        $vis type $KernelType = $fn_name;
    };
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis kernel $KernelType:ident fn $fn_name:ident (
            $($params:tt)*
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @translated
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            msl = $msl_body,
            wgsl = $wgsl_body,
            body = { $($body)* }
        }
        $vis type $KernelType = $fn_name;
    };
}

/// 可组合 `#[gpu]` 纯计算辅助函数（Device Helper Function）展开宏。
#[macro_export]
macro_rules! sumeru_gpu_helper_fn {
    (
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $( $p_ident:ident : [$p_mut:ident] $p_ty:ty => $p_kind:expr ),* $(,)?
        ) -> $ret_ty:ty => $ret_kind:expr,
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            #[allow(unused_mut)]
            pub fn call( $( mut $p_ident : $p_ty ),* ) -> $ret_ty {
                $($body)*
            }

            #[inline]
            pub fn gpu_helper_descriptor() -> &'static $crate::GpuHelperFnDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuHelperFnDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    $crate::GpuHelperFnDescriptor {
                        name: stringify!($fn_name),
                        params: ::std::vec![
                            $( (stringify!($p_ident), $crate::__sumeru_gpu_is_mut!([$p_mut]), $p_kind) ),*
                        ],
                        return_type: $ret_kind,
                        body: $ir_block,
                    }
                })
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = fn( $( $p_ty ),* ) -> $ret_ty;

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: fn( $( $p_ty ),* ) -> $ret_ty = $fn_name::call;
                &FN_PTR
            }
        }
    };
}
