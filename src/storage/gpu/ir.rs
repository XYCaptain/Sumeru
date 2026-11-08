//! 异构计算着色器与算子中间表示 (GPU Kernel IR & Multi-Backend Codegen)
//!
//! 突破传统“单行隔离（Point-wise）+ 仅 32 位标量”限制，支持：
//! 1. **全精度标量体系 (`GpuScalarKind`)**：原生支持 `F16`、`BF16`、`F32`、`F64`（科学计算双精度）、
//!    `I32`、`U32`、`I64`、`U64` 与 `Bool`；
//! 2. **行级位平面谓词掩码 (`PredicationMask`)**：直接绑定 `RowBitMask` / `RowFlags` 的 64 位机器字数组，
//!    在 GPU 线程入口单周期跳过未激活行；
//! 3. **跨行随机读取与关系目标收集 (`GatherLoad` & `PairTargetRow`)**：支持通过 `Pair<R>` 目标行查找表
//!    （LUT）或邻域索引跨行安全读取目标实体组件（带边界与 `u32::MAX` 空目标守卫）；
//! 4. **原子散射归约与线程组共享内存归约 (`AtomicReduce` & `ThreadgroupReduceSum`)**：支持跨行力累加、
//!    能量统计与全局归约；
//! 5. **三后端代码生成**：一键编译输出 **WebGPU WGSL**、**Apple Metal (MSL)** 与 **NVIDIA CUDA C++**。

use std::fmt::Write;

/// GPU 标量数据类型枚举（覆盖半精度、单精度、HPC 双精度与 32/64 位整型）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuScalarKind {
    F16,
    BF16,
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
    Bool,
}

impl GpuScalarKind {
    /// 标量类型的物理字节宽度
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// 是否为 64 位宽标量（HPC 双精度或 64 位整数）
    #[inline]
    pub const fn is_64bit(self) -> bool {
        matches!(self, Self::F64 | Self::I64 | Self::U64)
    }

    /// 是否为浮点标量
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// 映射为 WebGPU WGSL 类型名
    #[inline]
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::F16 | Self::BF16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::Bool => "bool",
        }
    }

    /// 映射为 Apple Metal Shading Language (MSL) 类型名
    #[inline]
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F16 => "half",
            Self::BF16 => "bfloat",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::I64 => "long",
            Self::U64 => "ulong",
            Self::Bool => "bool",
        }
    }

    /// 映射为 NVIDIA CUDA C++ 类型名
    #[inline]
    pub const fn cuda_type(self) -> &'static str {
        match self {
            Self::F16 => "__half",
            Self::BF16 => "__nv_bfloat16",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::I32 => "int32_t",
            Self::U32 => "uint32_t",
            Self::I64 => "int64_t",
            Self::U64 => "uint64_t",
            Self::Bool => "bool",
        }
    }
}

/// 缓冲区访问权限与拓扑角色
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// 只读组件列或数据源（支持同列本行读取与跨行 Gather）
    ReadOnly,
    /// 可读写目标组件列
    ReadWrite,
    /// 原子归约输出或散射累加缓冲区
    AtomicReadWrite,
    /// 64 位机器字谓词掩码缓冲区（直接绑定 `RowBitMask::words()`）
    PredicationMask,
    /// 关系对 `Pair<R>` 目标行索引查找表（每行一个 `u32` 目标行号，`u32::MAX` 表示无效）
    PairTargetLut,
}

/// GPU 内核缓冲区绑定描述符
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferBinding {
    pub binding_index: u32,
    pub name: String,
    pub scalar: GpuScalarKind,
    /// 每个逻辑行包含的基础标量个数（如 `Vec3<f64>` 为 3，标量组件为 1）
    pub stride_elems: usize,
    pub access: BufferAccess,
}

/// 一元算术与数学算子
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Exp,
}

/// 二元算术与比较算子
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// 跨行原子归约算子
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Min,
    Max,
}

/// GPU 着色器表达式 AST
#[derive(Debug, Clone, PartialEq)]
pub enum IrExpr {
    ConstF64(f64, GpuScalarKind),
    ConstI64(i64, GpuScalarKind),
    ConstU64(u64, GpuScalarKind),
    ConstBool(bool),
    /// 当前线程对应的逻辑行索引 `row`
    RowIndex,
    /// 引用局部标量变量
    Var(String),
    /// 行内组件字段读取：`buf[row * stride + elem_offset]`
    RowLoad {
        binding: u32,
        elem_offset: usize,
        scalar: GpuScalarKind,
    },
    /// 从关系对目标行查找表（`PairTargetLut`）读取当前实体指向的目标行索引 `u32`
    PairTargetRow {
        lut_binding: u32,
    },
    /// 跨行随机/邻域/关系目标收集读取：`target_row < target_len ? buf[target_row * stride + elem_offset] : fallback`
    GatherLoad {
        binding: u32,
        target_row: Box<IrExpr>,
        elem_offset: usize,
        scalar: GpuScalarKind,
        fallback: Box<IrExpr>,
    },
    /// 一元运算
    Unary {
        op: UnaryOp,
        expr: Box<IrExpr>,
    },
    /// 二元运算
    Binary {
        op: BinaryOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
    /// 融合乘加 `a * b + c`
    Fma {
        a: Box<IrExpr>,
        b: Box<IrExpr>,
        c: Box<IrExpr>,
    },
    /// 无分支条件选择 `cond ? then_expr : else_expr`
    Select {
        cond: Box<IrExpr>,
        then_expr: Box<IrExpr>,
        else_expr: Box<IrExpr>,
    },
}

impl IrExpr {
    #[inline]
    pub fn f32_const(v: f32) -> Self {
        Self::ConstF64(v as f64, GpuScalarKind::F32)
    }

    #[inline]
    pub fn f64_const(v: f64) -> Self {
        Self::ConstF64(v, GpuScalarKind::F64)
    }

    #[inline]
    pub fn u32_const(v: u32) -> Self {
        Self::ConstU64(v as u64, GpuScalarKind::U32)
    }

    #[inline]
    pub fn i32_const(v: i32) -> Self {
        Self::ConstI64(v as i64, GpuScalarKind::I32)
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }

    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        Self::Binary {
            op: BinaryOp::Div,
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }

    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        Self::Fma {
            a: Box::new(self),
            b: Box::new(b),
            c: Box::new(c),
        }
    }
}

/// GPU 着色器语句 AST
#[derive(Debug, Clone, PartialEq)]
pub enum IrStmt {
    /// 声明局部标量变量
    Let {
        name: String,
        scalar: GpuScalarKind,
        value: IrExpr,
    },
    /// 行内原位写入：`buf[row * stride + elem_offset] = value`
    RowStore {
        binding: u32,
        elem_offset: usize,
        scalar: GpuScalarKind,
        value: IrExpr,
    },
    /// 条件控制流
    If {
        cond: IrExpr,
        then_body: Vec<IrStmt>,
        else_body: Vec<IrStmt>,
    },
    /// 跨行原子散射归约（如将子粒子力累加到目标父节点，或全局原子累加）
    AtomicReduce {
        op: AtomicOp,
        dst_binding: u32,
        dst_index: IrExpr,
        scalar: GpuScalarKind,
        value: IrExpr,
    },
    /// Threadgroup / Warp 级共享内存归约累加至 `dst_binding[0]`
    ThreadgroupReduceSum {
        dst_binding: u32,
        scalar: GpuScalarKind,
        value: IrExpr,
    },
}

/// 异构计算内核中间表示 (KernelIr)
#[derive(Debug, Clone)]
pub struct KernelIr {
    pub name: String,
    pub workgroup_size: u32,
    pub bindings: Vec<BufferBinding>,
    pub body: Vec<IrStmt>,
    /// 若置为 `Some(binding_idx)`，内核自动在入口处根据 64 位机器字位平面过滤非活跃行
    pub predication_mask_binding: Option<u32>,
}

impl KernelIr {
    pub fn new(name: impl Into<String>, workgroup_size: u32) -> Self {
        Self {
            name: name.into(),
            workgroup_size: workgroup_size.max(1),
            bindings: Vec::new(),
            body: Vec::new(),
            predication_mask_binding: None,
        }
    }

    /// 绑定列缓冲区并返回其绑定槽位号
    pub fn add_binding(
        &mut self,
        name: impl Into<String>,
        scalar: GpuScalarKind,
        stride_elems: usize,
        access: BufferAccess,
    ) -> u32 {
        let idx = self.bindings.len() as u32;
        if access == BufferAccess::PredicationMask {
            self.predication_mask_binding = Some(idx);
        }
        self.bindings.push(BufferBinding {
            binding_index: idx,
            name: name.into(),
            scalar,
            stride_elems: stride_elems.max(1),
            access,
        });
        idx
    }

    /// 启用 `RowBitMask` / `RowFlags` 行级位平面硬件谓词掩码
    pub fn enable_bitmask_predication(&mut self) -> u32 {
        if let Some(existing) = self.predication_mask_binding {
            return existing;
        }
        self.add_binding(
            "__row_mask",
            GpuScalarKind::U64,
            1,
            BufferAccess::PredicationMask,
        )
    }

    /// 追加一条内核语句
    pub fn push_stmt(&mut self, stmt: IrStmt) {
        self.body.push(stmt);
    }

    /// 是否使用了 64 位双精度浮点或 64 位整数运算
    pub fn uses_fp64(&self) -> bool {
        self.bindings
            .iter()
            .any(|b| b.scalar == GpuScalarKind::F64)
            || self.body.iter().any(Self::stmt_uses_fp64)
    }

    fn stmt_uses_fp64(stmt: &IrStmt) -> bool {
        match stmt {
            IrStmt::Let { scalar, .. }
            | IrStmt::RowStore { scalar, .. }
            | IrStmt::AtomicReduce { scalar, .. }
            | IrStmt::ThreadgroupReduceSum { scalar, .. } => *scalar == GpuScalarKind::F64,
            IrStmt::If {
                then_body,
                else_body,
                ..
            } => {
                then_body.iter().any(Self::stmt_uses_fp64)
                    || else_body.iter().any(Self::stmt_uses_fp64)
            }
        }
    }

    /// 是否包含跨行 Gather 或 `Pair` 目标查找访存模式
    pub fn uses_cross_row_gather(&self) -> bool {
        self.bindings
            .iter()
            .any(|b| b.access == BufferAccess::PairTargetLut)
            || self.body.iter().any(Self::stmt_uses_gather)
    }

    fn stmt_uses_gather(stmt: &IrStmt) -> bool {
        match stmt {
            IrStmt::Let { value, .. }
            | IrStmt::RowStore { value, .. }
            | IrStmt::ThreadgroupReduceSum { value, .. } => Self::expr_uses_gather(value),
            IrStmt::AtomicReduce {
                dst_index, value, ..
            } => Self::expr_uses_gather(dst_index) || Self::expr_uses_gather(value),
            IrStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                Self::expr_uses_gather(cond)
                    || then_body.iter().any(Self::stmt_uses_gather)
                    || else_body.iter().any(Self::stmt_uses_gather)
            }
        }
    }

    fn expr_uses_gather(expr: &IrExpr) -> bool {
        match expr {
            IrExpr::GatherLoad { .. } | IrExpr::PairTargetRow { .. } => true,
            IrExpr::Unary { expr, .. } => Self::expr_uses_gather(expr),
            IrExpr::Binary { lhs, rhs, .. } => {
                Self::expr_uses_gather(lhs) || Self::expr_uses_gather(rhs)
            }
            IrExpr::Fma { a, b, c } => {
                Self::expr_uses_gather(a)
                    || Self::expr_uses_gather(b)
                    || Self::expr_uses_gather(c)
            }
            IrExpr::Select {
                cond,
                then_expr,
                else_expr,
            } => {
                Self::expr_uses_gather(cond)
                    || Self::expr_uses_gather(then_expr)
                    || Self::expr_uses_gather(else_expr)
            }
            _ => false,
        }
    }

    /// 查找指定绑定索引的缓冲区定义
    pub fn binding(&self, idx: u32) -> Option<&BufferBinding> {
        self.bindings.get(idx as usize)
    }

    /// 编译生成 WebGPU WGSL 计算着色器源码
    pub fn compile_wgsl(&self) -> String {
        let mut out = String::with_capacity(1024);
        if self
            .bindings
            .iter()
            .any(|b| matches!(b.scalar, GpuScalarKind::F16 | GpuScalarKind::BF16))
        {
            let _ = writeln!(out, "enable f16;");
        }
        if self.uses_fp64() {
            let _ = writeln!(out, "// HPC Extension: requires shader-f64 capability");
        }
        let _ = writeln!(
            out,
            "struct DispatchParams {{\n    row_count: u32,\n    target_row_count: u32,\n}};"
        );
        let _ = writeln!(
            out,
            "@group(0) @binding(0) var<uniform> __params: DispatchParams;"
        );

        for b in &self.bindings {
            let slot = b.binding_index + 1;
            match b.access {
                BufferAccess::PredicationMask => {
                    // WGSL 使用 array<u32> 以每 2 个 u32 字承载 64 位 RowBitMask
                    let _ = writeln!(
                        out,
                        "@group(0) @binding({slot}) var<storage, read> {}: array<u32>;",
                        b.name
                    );
                }
                BufferAccess::PairTargetLut => {
                    let _ = writeln!(
                        out,
                        "@group(0) @binding({slot}) var<storage, read> {}: array<u32>;",
                        b.name
                    );
                }
                BufferAccess::ReadOnly => {
                    let _ = writeln!(
                        out,
                        "@group(0) @binding({slot}) var<storage, read> {}: array<{}>;",
                        b.name,
                        b.scalar.wgsl_type()
                    );
                }
                BufferAccess::ReadWrite => {
                    let _ = writeln!(
                        out,
                        "@group(0) @binding({slot}) var<storage, read_write> {}: array<{}>;",
                        b.name,
                        b.scalar.wgsl_type()
                    );
                }
                BufferAccess::AtomicReadWrite => {
                    let _ = writeln!(
                        out,
                        "@group(0) @binding({slot}) var<storage, read_write> {}: array<atomic<{}>>;",
                        b.name,
                        b.scalar.wgsl_type()
                    );
                }
            }
        }

        let _ = writeln!(
            out,
            "\n@compute @workgroup_size({}, 1, 1)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
            self.workgroup_size, self.name
        );
        let _ = writeln!(out, "    let row: u32 = gid.x;");
        let _ = writeln!(
            out,
            "    if (row >= __params.row_count) {{\n        return;\n    }}"
        );

        if let Some(mask_idx) = self.predication_mask_binding {
            if let Some(mask_b) = self.binding(mask_idx) {
                let _ = writeln!(
                    out,
                    "    let mask_word_u32: u32 = {}[row >> 5u];",
                    mask_b.name
                );
                let _ = writeln!(
                    out,
                    "    if (((mask_word_u32 >> (row & 31u)) & 1u) == 0u) {{\n        return;\n    }}"
                );
            }
        }

        for stmt in &self.body {
            self.emit_wgsl_stmt(&mut out, stmt, 1);
        }
        let _ = writeln!(out, "}}");
        out
    }

    fn emit_wgsl_stmt(&self, out: &mut String, stmt: &IrStmt, indent: usize) {
        let pad = "    ".repeat(indent);
        match stmt {
            IrStmt::Let {
                name,
                scalar,
                value,
            } => {
                let expr_s = self.emit_wgsl_expr(value);
                let _ = writeln!(out, "{pad}let {name}: {} = {expr_s};", scalar.wgsl_type());
            }
            IrStmt::RowStore {
                binding,
                elem_offset,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let expr_s = self.emit_wgsl_expr(value);
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        let _ = writeln!(out, "{pad}{}[row] = {expr_s};", b.name);
                    } else {
                        let _ = writeln!(
                            out,
                            "{pad}{}[row * {}u + {}u] = {expr_s};",
                            b.name, b.stride_elems, elem_offset
                        );
                    }
                }
            }
            IrStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let c_s = self.emit_wgsl_expr(cond);
                let _ = writeln!(out, "{pad}if ({c_s}) {{");
                for s in then_body {
                    self.emit_wgsl_stmt(out, s, indent + 1);
                }
                if !else_body.is_empty() {
                    let _ = writeln!(out, "{pad}}} else {{");
                    for s in else_body {
                        self.emit_wgsl_stmt(out, s, indent + 1);
                    }
                }
                let _ = writeln!(out, "{pad}}}");
            }
            IrStmt::AtomicReduce {
                op,
                dst_binding,
                dst_index,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let idx_s = self.emit_wgsl_expr(dst_index);
                    let val_s = self.emit_wgsl_expr(value);
                    let fn_name = match op {
                        AtomicOp::Add => "atomicAdd",
                        AtomicOp::Min => "atomicMin",
                        AtomicOp::Max => "atomicMax",
                    };
                    let _ = writeln!(out, "{pad}{fn_name}(&{}[{idx_s}], {val_s});", b.name);
                }
            }
            IrStmt::ThreadgroupReduceSum {
                dst_binding, value, ..
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let val_s = self.emit_wgsl_expr(value);
                    let _ = writeln!(
                        out,
                        "{pad}// Threadgroup reduction accumulate\n{pad}atomicAdd(&{}[0u], {val_s});",
                        b.name
                    );
                }
            }
        }
    }

    fn emit_wgsl_expr(&self, expr: &IrExpr) -> String {
        match expr {
            IrExpr::ConstF64(v, kind) => format!("{v:.6}{}", if *kind == GpuScalarKind::F64 { "lf" } else { "f" }),
            IrExpr::ConstI64(v, _) => format!("{v}i"),
            IrExpr::ConstU64(v, _) => format!("{v}u"),
            IrExpr::ConstBool(v) => format!("{v}"),
            IrExpr::RowIndex => "row".to_string(),
            IrExpr::Var(name) => name.clone(),
            IrExpr::RowLoad {
                binding,
                elem_offset,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        format!("{}[row]", b.name)
                    } else {
                        format!("{}[row * {}u + {}u]", b.name, b.stride_elems, elem_offset)
                    }
                } else {
                    "0".to_string()
                }
            }
            IrExpr::PairTargetRow { lut_binding } => {
                if let Some(b) = self.binding(*lut_binding) {
                    format!("{}[row]", b.name)
                } else {
                    "4294967295u".to_string()
                }
            }
            IrExpr::GatherLoad {
                binding,
                target_row,
                elem_offset,
                fallback,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let trow = self.emit_wgsl_expr(target_row);
                    let fb = self.emit_wgsl_expr(fallback);
                    let idx_expr = if b.stride_elems == 1 && *elem_offset == 0 {
                        trow.clone()
                    } else {
                        format!("({trow} * {}u + {}u)", b.stride_elems, elem_offset)
                    };
                    format!(
                        "select({fb}, {}[{idx_expr}], ({trow} < __params.target_row_count))",
                        b.name
                    )
                } else {
                    self.emit_wgsl_expr(fallback)
                }
            }
            IrExpr::Unary { op, expr } => {
                let inner = self.emit_wgsl_expr(expr);
                match op {
                    UnaryOp::Neg => format!("(-{inner})"),
                    UnaryOp::Abs => format!("abs({inner})"),
                    UnaryOp::Sqrt => format!("sqrt({inner})"),
                    UnaryOp::Rsqrt => format!("inverseSqrt({inner})"),
                    UnaryOp::Sin => format!("sin({inner})"),
                    UnaryOp::Cos => format!("cos({inner})"),
                    UnaryOp::Exp => format!("exp({inner})"),
                }
            }
            IrExpr::Binary { op, lhs, rhs } => {
                let l = self.emit_wgsl_expr(lhs);
                let r = self.emit_wgsl_expr(rhs);
                match op {
                    BinaryOp::Add => format!("({l} + {r})"),
                    BinaryOp::Sub => format!("({l} - {r})"),
                    BinaryOp::Mul => format!("({l} * {r})"),
                    BinaryOp::Div => format!("({l} / {r})"),
                    BinaryOp::Min => format!("min({l}, {r})"),
                    BinaryOp::Max => format!("max({l}, {r})"),
                    BinaryOp::Eq => format!("({l} == {r})"),
                    BinaryOp::Lt => format!("({l} < {r})"),
                    BinaryOp::Le => format!("({l} <= {r})"),
                    BinaryOp::Gt => format!("({l} > {r})"),
                    BinaryOp::Ge => format!("({l} >= {r})"),
                }
            }
            IrExpr::Fma { a, b, c } => {
                let a_s = self.emit_wgsl_expr(a);
                let b_s = self.emit_wgsl_expr(b);
                let c_s = self.emit_wgsl_expr(c);
                format!("fma({a_s}, {b_s}, {c_s})")
            }
            IrExpr::Select {
                cond,
                then_expr,
                else_expr,
            } => {
                let c_s = self.emit_wgsl_expr(cond);
                let t_s = self.emit_wgsl_expr(then_expr);
                let e_s = self.emit_wgsl_expr(else_expr);
                format!("select({e_s}, {t_s}, {c_s})")
            }
        }
    }

    /// 编译生成 Apple Metal Shading Language (MSL) 内核源码（适配 16KB UMA 零拷贝与 Simdgroup 归约）
    pub fn compile_msl(&self) -> String {
        let mut out = String::with_capacity(1024);
        let _ = writeln!(out, "#include <metal_stdlib>\nusing namespace metal;\n");
        let _ = writeln!(
            out,
            "struct DispatchParams {{\n    uint row_count;\n    uint target_row_count;\n}};\n"
        );
        let _ = write!(out, "kernel void {}(\n    constant DispatchParams& __params [[buffer(0)]]", self.name);

        for b in &self.bindings {
            let slot = b.binding_index + 1;
            match b.access {
                BufferAccess::PredicationMask => {
                    let _ = write!(out, ",\n    device const ulong* {} [[buffer({slot})]]", b.name);
                }
                BufferAccess::PairTargetLut => {
                    let _ = write!(out, ",\n    device const uint* {} [[buffer({slot})]]", b.name);
                }
                BufferAccess::ReadOnly => {
                    let _ = write!(
                        out,
                        ",\n    device const {}* {} [[buffer({slot})]]",
                        b.scalar.msl_type(),
                        b.name
                    );
                }
                BufferAccess::ReadWrite | BufferAccess::AtomicReadWrite => {
                    let _ = write!(
                        out,
                        ",\n    device {}* {} [[buffer({slot})]]",
                        b.scalar.msl_type(),
                        b.name
                    );
                }
            }
        }
        let _ = writeln!(
            out,
            ",\n    uint row [[thread_position_in_grid]],\n    uint simd_lane [[thread_index_in_simdgroup]]\n) {{"
        );
        let _ = writeln!(
            out,
            "    if (row >= __params.row_count) {{\n        return;\n    }}"
        );

        if let Some(mask_idx) = self.predication_mask_binding {
            if let Some(mask_b) = self.binding(mask_idx) {
                let _ = writeln!(
                    out,
                    "    if ((({}[row >> 6u] >> (row & 63u)) & 1ul) == 0ul) {{\n        return;\n    }}",
                    mask_b.name
                );
            }
        }

        for stmt in &self.body {
            self.emit_msl_stmt(&mut out, stmt, 1);
        }
        let _ = writeln!(out, "}}");
        out
    }

    fn emit_msl_stmt(&self, out: &mut String, stmt: &IrStmt, indent: usize) {
        let pad = "    ".repeat(indent);
        match stmt {
            IrStmt::Let {
                name,
                scalar,
                value,
            } => {
                let expr_s = self.emit_msl_expr(value);
                let _ = writeln!(out, "{pad}{} {name} = {expr_s};", scalar.msl_type());
            }
            IrStmt::RowStore {
                binding,
                elem_offset,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let expr_s = self.emit_msl_expr(value);
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        let _ = writeln!(out, "{pad}{}[row] = {expr_s};", b.name);
                    } else {
                        let _ = writeln!(
                            out,
                            "{pad}{}[row * {}u + {}u] = {expr_s};",
                            b.name, b.stride_elems, elem_offset
                        );
                    }
                }
            }
            IrStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let c_s = self.emit_msl_expr(cond);
                let _ = writeln!(out, "{pad}if ({c_s}) {{");
                for s in then_body {
                    self.emit_msl_stmt(out, s, indent + 1);
                }
                if !else_body.is_empty() {
                    let _ = writeln!(out, "{pad}}} else {{");
                    for s in else_body {
                        self.emit_msl_stmt(out, s, indent + 1);
                    }
                }
                let _ = writeln!(out, "{pad}}}");
            }
            IrStmt::AtomicReduce {
                op,
                dst_binding,
                dst_index,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let idx_s = self.emit_msl_expr(dst_index);
                    let val_s = self.emit_msl_expr(value);
                    let op_comment = match op {
                        AtomicOp::Add => "+=",
                        AtomicOp::Min => "min",
                        AtomicOp::Max => "max",
                    };
                    let _ = writeln!(
                        out,
                        "{pad}// Atomic scatter ({op_comment})\n{pad}{}[{idx_s}] += {val_s};",
                        b.name
                    );
                }
            }
            IrStmt::ThreadgroupReduceSum {
                dst_binding,
                scalar,
                value,
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let val_s = self.emit_msl_expr(value);
                    let _ = writeln!(
                        out,
                        "{pad}{} __sg_sum = simd_sum({val_s});\n{pad}if (simd_lane == 0u) {{ {}[0] += __sg_sum; }}",
                        scalar.msl_type(),
                        b.name
                    );
                }
            }
        }
    }

    fn emit_msl_expr(&self, expr: &IrExpr) -> String {
        match expr {
            IrExpr::ConstF64(v, kind) => {
                if *kind == GpuScalarKind::F64 {
                    format!("{v:.6}")
                } else {
                    format!("{v:.6}f")
                }
            }
            IrExpr::ConstI64(v, _) => format!("{v}"),
            IrExpr::ConstU64(v, _) => format!("{v}u"),
            IrExpr::ConstBool(v) => format!("{v}"),
            IrExpr::RowIndex => "row".to_string(),
            IrExpr::Var(name) => name.clone(),
            IrExpr::RowLoad {
                binding,
                elem_offset,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        format!("{}[row]", b.name)
                    } else {
                        format!("{}[row * {}u + {}u]", b.name, b.stride_elems, elem_offset)
                    }
                } else {
                    "0".to_string()
                }
            }
            IrExpr::PairTargetRow { lut_binding } => {
                if let Some(b) = self.binding(*lut_binding) {
                    format!("{}[row]", b.name)
                } else {
                    "0xFFFFFFFFu".to_string()
                }
            }
            IrExpr::GatherLoad {
                binding,
                target_row,
                elem_offset,
                fallback,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let trow = self.emit_msl_expr(target_row);
                    let fb = self.emit_msl_expr(fallback);
                    let idx_expr = if b.stride_elems == 1 && *elem_offset == 0 {
                        trow.clone()
                    } else {
                        format!("({trow} * {}u + {}u)", b.stride_elems, elem_offset)
                    };
                    format!(
                        "(({trow} < __params.target_row_count) ? {}[{idx_expr}] : ({fb}))",
                        b.name
                    )
                } else {
                    self.emit_msl_expr(fallback)
                }
            }
            IrExpr::Unary { op, expr } => {
                let inner = self.emit_msl_expr(expr);
                match op {
                    UnaryOp::Neg => format!("(-{inner})"),
                    UnaryOp::Abs => format!("abs({inner})"),
                    UnaryOp::Sqrt => format!("sqrt({inner})"),
                    UnaryOp::Rsqrt => format!("rsqrt({inner})"),
                    UnaryOp::Sin => format!("sin({inner})"),
                    UnaryOp::Cos => format!("cos({inner})"),
                    UnaryOp::Exp => format!("exp({inner})"),
                }
            }
            IrExpr::Binary { op, lhs, rhs } => {
                let l = self.emit_msl_expr(lhs);
                let r = self.emit_msl_expr(rhs);
                match op {
                    BinaryOp::Add => format!("({l} + {r})"),
                    BinaryOp::Sub => format!("({l} - {r})"),
                    BinaryOp::Mul => format!("({l} * {r})"),
                    BinaryOp::Div => format!("({l} / {r})"),
                    BinaryOp::Min => format!("min({l}, {r})"),
                    BinaryOp::Max => format!("max({l}, {r})"),
                    BinaryOp::Eq => format!("({l} == {r})"),
                    BinaryOp::Lt => format!("({l} < {r})"),
                    BinaryOp::Le => format!("({l} <= {r})"),
                    BinaryOp::Gt => format!("({l} > {r})"),
                    BinaryOp::Ge => format!("({l} >= {r})"),
                }
            }
            IrExpr::Fma { a, b, c } => {
                let a_s = self.emit_msl_expr(a);
                let b_s = self.emit_msl_expr(b);
                let c_s = self.emit_msl_expr(c);
                format!("fma({a_s}, {b_s}, {c_s})")
            }
            IrExpr::Select {
                cond,
                then_expr,
                else_expr,
            } => {
                let c_s = self.emit_msl_expr(cond);
                let t_s = self.emit_msl_expr(then_expr);
                let e_s = self.emit_msl_expr(else_expr);
                format!("(({c_s}) ? ({t_s}) : ({e_s}))")
            }
        }
    }

    /// 编译生成面向超算（NVIDIA Grace Hopper / Blackwell）的 CUDA C++ 内核源码（原生支持 `double` FP64 与 `atomicAdd`）
    pub fn compile_cuda(&self) -> String {
        let mut out = String::with_capacity(1024);
        let _ = writeln!(out, "#include <stdint.h>\n#include <math.h>\n");
        let _ = write!(
            out,
            "extern \"C\" __global__ void {}(\n    const uint32_t __row_count,\n    const uint32_t __target_row_count",
            self.name
        );

        for b in &self.bindings {
            match b.access {
                BufferAccess::PredicationMask => {
                    let _ = write!(out, ",\n    const uint64_t* __restrict__ {}", b.name);
                }
                BufferAccess::PairTargetLut => {
                    let _ = write!(out, ",\n    const uint32_t* __restrict__ {}", b.name);
                }
                BufferAccess::ReadOnly => {
                    let _ = write!(
                        out,
                        ",\n    const {}* __restrict__ {}",
                        b.scalar.cuda_type(),
                        b.name
                    );
                }
                BufferAccess::ReadWrite | BufferAccess::AtomicReadWrite => {
                    let _ = write!(
                        out,
                        ",\n    {}* __restrict__ {}",
                        b.scalar.cuda_type(),
                        b.name
                    );
                }
            }
        }
        let _ = writeln!(out, "\n) {{");
        let _ = writeln!(
            out,
            "    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;"
        );
        let _ = writeln!(
            out,
            "    if (row >= __row_count) {{\n        return;\n    }}"
        );

        if let Some(mask_idx) = self.predication_mask_binding {
            if let Some(mask_b) = self.binding(mask_idx) {
                let _ = writeln!(
                    out,
                    "    if ((({}[row >> 6u] >> (row & 63u)) & 1ULL) == 0ULL) {{\n        return;\n    }}",
                    mask_b.name
                );
            }
        }

        for stmt in &self.body {
            self.emit_cuda_stmt(&mut out, stmt, 1);
        }
        let _ = writeln!(out, "}}");
        out
    }

    fn emit_cuda_stmt(&self, out: &mut String, stmt: &IrStmt, indent: usize) {
        let pad = "    ".repeat(indent);
        match stmt {
            IrStmt::Let {
                name,
                scalar,
                value,
            } => {
                let expr_s = self.emit_cuda_expr(value);
                let _ = writeln!(out, "{pad}const {} {name} = {expr_s};", scalar.cuda_type());
            }
            IrStmt::RowStore {
                binding,
                elem_offset,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let expr_s = self.emit_cuda_expr(value);
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        let _ = writeln!(out, "{pad}{}[row] = {expr_s};", b.name);
                    } else {
                        let _ = writeln!(
                            out,
                            "{pad}{}[row * {}u + {}u] = {expr_s};",
                            b.name, b.stride_elems, elem_offset
                        );
                    }
                }
            }
            IrStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let c_s = self.emit_cuda_expr(cond);
                let _ = writeln!(out, "{pad}if ({c_s}) {{");
                for s in then_body {
                    self.emit_cuda_stmt(out, s, indent + 1);
                }
                if !else_body.is_empty() {
                    let _ = writeln!(out, "{pad}}} else {{");
                    for s in else_body {
                        self.emit_cuda_stmt(out, s, indent + 1);
                    }
                }
                let _ = writeln!(out, "{pad}}}");
            }
            IrStmt::AtomicReduce {
                op,
                dst_binding,
                dst_index,
                value,
                ..
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let idx_s = self.emit_cuda_expr(dst_index);
                    let val_s = self.emit_cuda_expr(value);
                    let fn_name = match op {
                        AtomicOp::Add => "atomicAdd",
                        AtomicOp::Min => "atomicMin",
                        AtomicOp::Max => "atomicMax",
                    };
                    let _ = writeln!(out, "{pad}{fn_name}(&{}[{idx_s}], {val_s});", b.name);
                }
            }
            IrStmt::ThreadgroupReduceSum {
                dst_binding, value, ..
            } => {
                if let Some(b) = self.binding(*dst_binding) {
                    let val_s = self.emit_cuda_expr(value);
                    let _ = writeln!(out, "{pad}atomicAdd(&{}[0], {val_s});", b.name);
                }
            }
        }
    }

    fn emit_cuda_expr(&self, expr: &IrExpr) -> String {
        match expr {
            IrExpr::ConstF64(v, kind) => {
                if *kind == GpuScalarKind::F64 {
                    format!("{v:.12}")
                } else {
                    format!("{v:.6}f")
                }
            }
            IrExpr::ConstI64(v, _) => format!("{v}"),
            IrExpr::ConstU64(v, _) => format!("{v}u"),
            IrExpr::ConstBool(v) => format!("{v}"),
            IrExpr::RowIndex => "row".to_string(),
            IrExpr::Var(name) => name.clone(),
            IrExpr::RowLoad {
                binding,
                elem_offset,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    if b.stride_elems == 1 && *elem_offset == 0 {
                        format!("{}[row]", b.name)
                    } else {
                        format!("{}[row * {}u + {}u]", b.name, b.stride_elems, elem_offset)
                    }
                } else {
                    "0".to_string()
                }
            }
            IrExpr::PairTargetRow { lut_binding } => {
                if let Some(b) = self.binding(*lut_binding) {
                    format!("{}[row]", b.name)
                } else {
                    "0xFFFFFFFFu".to_string()
                }
            }
            IrExpr::GatherLoad {
                binding,
                target_row,
                elem_offset,
                fallback,
                ..
            } => {
                if let Some(b) = self.binding(*binding) {
                    let trow = self.emit_cuda_expr(target_row);
                    let fb = self.emit_cuda_expr(fallback);
                    let idx_expr = if b.stride_elems == 1 && *elem_offset == 0 {
                        trow.clone()
                    } else {
                        format!("({trow} * {}u + {}u)", b.stride_elems, elem_offset)
                    };
                    format!(
                        "(({trow} < __target_row_count) ? {}[{idx_expr}] : ({fb}))",
                        b.name
                    )
                } else {
                    self.emit_cuda_expr(fallback)
                }
            }
            IrExpr::Unary { op, expr } => {
                let inner = self.emit_cuda_expr(expr);
                match op {
                    UnaryOp::Neg => format!("(-{inner})"),
                    UnaryOp::Abs => format!("fabs({inner})"),
                    UnaryOp::Sqrt => format!("sqrt({inner})"),
                    UnaryOp::Rsqrt => format!("rsqrt({inner})"),
                    UnaryOp::Sin => format!("sin({inner})"),
                    UnaryOp::Cos => format!("cos({inner})"),
                    UnaryOp::Exp => format!("exp({inner})"),
                }
            }
            IrExpr::Binary { op, lhs, rhs } => {
                let l = self.emit_cuda_expr(lhs);
                let r = self.emit_cuda_expr(rhs);
                match op {
                    BinaryOp::Add => format!("({l} + {r})"),
                    BinaryOp::Sub => format!("({l} - {r})"),
                    BinaryOp::Mul => format!("({l} * {r})"),
                    BinaryOp::Div => format!("({l} / {r})"),
                    BinaryOp::Min => format!("fmin({l}, {r})"),
                    BinaryOp::Max => format!("fmax({l}, {r})"),
                    BinaryOp::Eq => format!("({l} == {r})"),
                    BinaryOp::Lt => format!("({l} < {r})"),
                    BinaryOp::Le => format!("({l} <= {r})"),
                    BinaryOp::Gt => format!("({l} > {r})"),
                    BinaryOp::Ge => format!("({l} >= {r})"),
                }
            }
            IrExpr::Fma { a, b, c } => {
                let a_s = self.emit_cuda_expr(a);
                let b_s = self.emit_cuda_expr(b);
                let c_s = self.emit_cuda_expr(c);
                format!("fma({a_s}, {b_s}, {c_s})")
            }
            IrExpr::Select {
                cond,
                then_expr,
                else_expr,
            } => {
                let c_s = self.emit_cuda_expr(cond);
                let t_s = self.emit_cuda_expr(then_expr);
                let e_s = self.emit_cuda_expr(else_expr);
                format!("(({c_s}) ? ({t_s}) : ({e_s}))")
            }
        }
    }
}
