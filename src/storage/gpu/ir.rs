//! 强类型着色器中间表示 (Typed Shader IR)、符号表类型推导与多后端代码生成器 (`GpuIrCompiler`)
//!
//! 本模块提供目标无关的着色器 AST/IR 树（[`GpuBlock`]、[`GpuStmt`]、[`GpuExpr`]）：
//! 1. **零字符串模板宏展开**：`crates/sumeru-macros` 仅负责将 Rust `syn` AST 展开为本模块的强类型 IR 构造调用；
//! 2. **借力 `rustc` 编译期常量求值 ([`GpuConstValue`])**：核函数内引用的外部 `const` 常量（如 `DT`、`std::f32::consts::PI`）
//!    直接由 Rust 编译器求值并内联为强类型字面量 [`GpuLit`]；
//! 3. **可组合 `#[gpu]` 纯计算辅助函数 ([`GpuHelperFnDescriptor`])**：支持定义带返回值的纯计算设备辅助函数，
//!    自动按调用依赖拓扑序链接至目标 WGSL 与 MSL 模块；
//! 4. **符号表驱动的类型推导 ([`ShaderTypeEnv`])**：结合 [`GpuShaderStruct::FIELDS`] 元数据自动推导组件字段、
//!    数组下标、局部变量与 `for i in 0..cfg.steps` 循环边界的有符号/无符号类型。

use super::kernel::{GpuFieldType, GpuKernelBindingSpec, GpuStructField};
use std::collections::HashMap;

/// 着色器标量基础类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuScalarKind {
    F32,
    I32,
    U32,
    Bool,
}

impl GpuScalarKind {
    /// 映射为 Apple Metal Shading Language (MSL) 标量类型名。
    #[inline]
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::Bool => "bool",
        }
    }

    /// 映射为 WebGPU Shading Language (WGSL) 标量类型名。
    #[inline]
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Bool => "bool",
        }
    }
}

/// 符号表推导出的表达式值类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuValueType {
    Scalar(GpuScalarKind),
    Array(GpuScalarKind, usize),
    Struct(&'static str),
    Unknown,
}

impl GpuValueType {
    /// 从组件结构体字段类型 [`GpuFieldType`] 转换为 [`GpuValueType`]。
    #[inline]
    pub const fn from_field_type(ft: GpuFieldType) -> Self {
        match ft {
            GpuFieldType::F32 => Self::Scalar(GpuScalarKind::F32),
            GpuFieldType::I32 => Self::Scalar(GpuScalarKind::I32),
            GpuFieldType::U32 => Self::Scalar(GpuScalarKind::U32),
            GpuFieldType::Vec2F32 => Self::Array(GpuScalarKind::F32, 2),
            GpuFieldType::Vec3F32 => Self::Array(GpuScalarKind::F32, 3),
            GpuFieldType::Vec4F32 => Self::Array(GpuScalarKind::F32, 4),
            GpuFieldType::Vec2U32 => Self::Array(GpuScalarKind::U32, 2),
            GpuFieldType::Vec3U32 => Self::Array(GpuScalarKind::U32, 3),
            GpuFieldType::Vec4U32 => Self::Array(GpuScalarKind::U32, 4),
            GpuFieldType::Vec2I32 => Self::Array(GpuScalarKind::I32, 2),
            GpuFieldType::Vec3I32 => Self::Array(GpuScalarKind::I32, 3),
            GpuFieldType::Vec4I32 => Self::Array(GpuScalarKind::I32, 4),
        }
    }
}

/// 强类型着色器字面量节点。
#[derive(Debug, Clone, PartialEq)]
pub enum GpuLit {
    /// 源码内联浮点字面量（保留原始十进制文本，如 `"100.0"`、`"1e-5"`）
    FloatRaw(&'static str),
    /// 由外部 Rust `const` 常量求值得到的 `f32` 字面量
    F32(f32),
    /// 源码内未带类型后缀的整数字面量（在 `U32` 上下文中可自动提升为 `u` 后缀）
    UntypedInt(i64),
    /// 显式有符号 32 位整数字面量 (`i32`)
    I32(i32),
    /// 显式无符号 32 位整数字面量 (`u32` / `usize`)
    U32(u64),
    /// 布尔字面量 (`bool`)
    Bool(bool),
}

impl GpuLit {
    /// 格式化为着色器源码字面量文本。
    pub fn format_shader(&self, coerce_unsigned: bool) -> String {
        match self {
            Self::FloatRaw(s) => (*s).to_string(),
            Self::F32(v) => {
                assert!(
                    v.is_finite(),
                    "Non-finite f32 constant `{v}` cannot be embedded into a GPU shader"
                );
                let s = format!("{v:?}");
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    s
                } else {
                    format!("{s}.0")
                }
            }
            Self::UntypedInt(n) => {
                if coerce_unsigned && *n >= 0 {
                    format!("{n}u")
                } else {
                    format!("{n}")
                }
            }
            Self::I32(n) => format!("{n}"),
            Self::U32(n) => format!("{n}u"),
            Self::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        }
    }
}

/// 将外部 Rust `const` 常量桥接为着色器 IR 表达式 [`GpuExpr`] 的编译期转换契约。
pub trait GpuConstValue {
    /// 将当前 Rust 常量值转换为强类型着色器字面量表达式。
    fn to_gpu_expr(&self) -> GpuExpr;
}

impl GpuConstValue for f32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::F32(*self))
    }
}

impl GpuConstValue for i32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::I32(*self))
    }
}

impl GpuConstValue for u32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::U32(*self as u64))
    }
}

impl GpuConstValue for usize {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::U32(*self as u64))
    }
}

impl GpuConstValue for bool {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::Bool(*self))
    }
}

/// 着色器二元运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
}

impl GpuBinaryOp {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
            Self::And => "&&",
            Self::Or => "||",
            Self::BitXor => "^",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Eq => "==",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Ne => "!=",
            Self::Ge => ">=",
            Self::Gt => ">",
        }
    }

    #[inline]
    pub const fn is_comparison_or_logical(self) -> bool {
        matches!(
            self,
            Self::And
                | Self::Or
                | Self::Eq
                | Self::Lt
                | Self::Le
                | Self::Ne
                | Self::Ge
                | Self::Gt
        )
    }
}

/// 着色器赋值与复合赋值运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitXorAssign,
    BitAndAssign,
    BitOrAssign,
    ShlAssign,
    ShrAssign,
}

impl GpuAssignOp {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assign => "=",
            Self::AddAssign => "+=",
            Self::SubAssign => "-=",
            Self::MulAssign => "*=",
            Self::DivAssign => "/=",
            Self::RemAssign => "%=",
            Self::BitXorAssign => "^=",
            Self::BitAndAssign => "&=",
            Self::BitOrAssign => "|=",
            Self::ShlAssign => "<<=",
            Self::ShrAssign => ">>=",
        }
    }
}

/// 着色器一元运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuUnaryOp {
    Neg,
    Not,
}

/// 跨平台着色器内建数学函数枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMathIntrinsic {
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    Fract,
    Exp,
    Ln,
    Min,
    Max,
    Atan2,
    Powf,
    Clamp,
    Fma,
}

/// 左值表达式节点（支持标识符、结构体字段访问与数组下标索引）。
#[derive(Debug, Clone)]
pub enum GpuLValue {
    Var(&'static str),
    Field {
        base: Box<GpuLValue>,
        field: &'static str,
    },
    Index {
        base: Box<GpuLValue>,
        index: Box<GpuExpr>,
    },
}

/// 可组合 `#[gpu]` 纯计算辅助函数（Device Function）静态描述符。
#[derive(Debug, Clone)]
pub struct GpuHelperFnDescriptor {
    /// 辅助函数名称
    pub name: &'static str,
    /// 形参列表 `(param_name, is_mut, scalar_kind)`
    pub params: Vec<(&'static str, bool, GpuScalarKind)>,
    /// 返回值标量类型
    pub return_type: GpuScalarKind,
    /// 函数体强类型 IR 块
    pub body: GpuBlock,
}

/// 强类型着色器右值表达式树节点。
#[derive(Debug, Clone)]
pub enum GpuExpr {
    /// 局部/形参变量引用
    Var(&'static str),
    /// 强类型字面量（含编译期求值的外部 `const` 常量）
    Lit(GpuLit),
    /// 结构体字段读取 (`base.field`)
    Field {
        base: Box<GpuExpr>,
        field: &'static str,
    },
    /// 数组元素下标读取 (`base[index]`)
    Index {
        base: Box<GpuExpr>,
        index: Box<GpuExpr>,
    },
    /// 二元运算表达式 (`lhs op rhs`)
    Binary {
        op: GpuBinaryOp,
        lhs: Box<GpuExpr>,
        rhs: Box<GpuExpr>,
    },
    /// 一元运算表达式 (`-expr` / `!expr`)
    Unary {
        op: GpuUnaryOp,
        expr: Box<GpuExpr>,
    },
    /// 括号分组表达式 (`(expr)`)
    Paren(Box<GpuExpr>),
    /// 标量类型强转 (`expr as ty`)
    Cast {
        expr: Box<GpuExpr>,
        ty: GpuScalarKind,
    },
    /// 跨平台数学内建函数调用
    MathCall {
        receiver: Box<GpuExpr>,
        intrinsic: GpuMathIntrinsic,
        args: Vec<GpuExpr>,
    },
    /// 调用另一个 `#[gpu]` 纯计算辅助函数
    HelperCall {
        helper: &'static GpuHelperFnDescriptor,
        args: Vec<GpuExpr>,
    },
}

/// 强类型着色器语句节点。
#[derive(Debug, Clone)]
pub enum GpuStmt {
    /// 局部变量声明 (`let [mut] name [: ty] = init;`)
    Let {
        name: &'static str,
        mutable: bool,
        explicit_ty: Option<GpuScalarKind>,
        init: GpuExpr,
    },
    /// 赋值或复合赋值语句 (`target op= value;`)
    Assign {
        op: GpuAssignOp,
        target: GpuLValue,
        value: GpuExpr,
    },
    /// 条件分支语句 (`if (cond) { then } else { ... }`)
    If {
        cond: GpuExpr,
        then_block: GpuBlock,
        else_branch: Option<GpuElseBranch>,
    },
    /// `while` 条件循环语句
    While {
        cond: GpuExpr,
        body: GpuBlock,
    },
    /// `for var in start..end` 范围循环语句
    ForRange {
        var: &'static str,
        start: GpuExpr,
        end: GpuExpr,
        inclusive: bool,
        body: GpuBlock,
    },
    /// 嵌套代码块
    Block(GpuBlock),
    /// 函数返回语句（核函数为 `None`，纯辅助函数为 `Some(expr)`）
    Return(Option<GpuExpr>),
    /// 循环跳出 (`break;`)
    Break,
    /// 循环继续 (`continue;`)
    Continue,
}

/// `if` 语句的 `else` 分支形态。
#[derive(Debug, Clone)]
pub enum GpuElseBranch {
    ElseIf(Box<GpuStmt>),
    ElseBlock(GpuBlock),
}

/// 着色器语句块。
#[derive(Debug, Clone, Default)]
pub struct GpuBlock {
    pub stmts: Vec<GpuStmt>,
}

impl GpuBlock {
    #[inline]
    pub fn new(stmts: Vec<GpuStmt>) -> Self {
        Self { stmts }
    }
}

/// 组件结构体 Schema 符号项（供 [`ShaderTypeEnv`] 查询字段类型）。
#[derive(Debug, Clone, Copy)]
pub struct GpuComponentSchema {
    pub param_name: &'static str,
    pub struct_name: &'static str,
    pub fields: &'static [GpuStructField],
    pub mutable: bool,
}

/// 词法作用域与组件字段类型推导环境。
#[derive(Clone)]
struct ShaderTypeEnv<'a> {
    schemas: HashMap<&'static str, &'static [GpuStructField]>,
    scopes: Vec<HashMap<&'static str, GpuValueType>>,
    mut_params: &'a [&'static str],
}

impl<'a> ShaderTypeEnv<'a> {
    fn for_kernel(components: &[GpuComponentSchema], mut_params: &'a [&'static str]) -> Self {
        let mut schemas = HashMap::with_capacity(components.len());
        let mut root_scope = HashMap::with_capacity(components.len());
        for c in components {
            schemas.insert(c.struct_name, c.fields);
            root_scope.insert(c.param_name, GpuValueType::Struct(c.struct_name));
        }
        Self {
            schemas,
            scopes: vec![root_scope],
            mut_params,
        }
    }

    fn for_helper(params: &[(&'static str, bool, GpuScalarKind)]) -> Self {
        let mut root_scope = HashMap::with_capacity(params.len());
        for &(name, _, kind) in params {
            root_scope.insert(name, GpuValueType::Scalar(kind));
        }
        Self {
            schemas: HashMap::new(),
            scopes: vec![root_scope],
            mut_params: &[],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &'static str, ty: GpuValueType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_var(&self, name: &str) -> GpuValueType {
        for scope in self.scopes.iter().rev() {
            if let Some(&ty) = scope.get(name) {
                return ty;
            }
        }
        GpuValueType::Unknown
    }

    fn lookup_field(&self, struct_name: &str, field_name: &str) -> GpuValueType {
        if let Some(&fields) = self.schemas.get(struct_name) {
            if let Some(f) = fields.iter().find(|f| f.name == field_name) {
                return GpuValueType::from_field_type(f.field_type);
            }
        }
        GpuValueType::Unknown
    }

    fn infer_expr_type(&self, expr: &GpuExpr) -> GpuValueType {
        match expr {
            GpuExpr::Var(name) => self.lookup_var(name),
            GpuExpr::Lit(lit) => match lit {
                GpuLit::FloatRaw(_) | GpuLit::F32(_) => GpuValueType::Scalar(GpuScalarKind::F32),
                GpuLit::UntypedInt(_) | GpuLit::I32(_) => GpuValueType::Scalar(GpuScalarKind::I32),
                GpuLit::U32(_) => GpuValueType::Scalar(GpuScalarKind::U32),
                GpuLit::Bool(_) => GpuValueType::Scalar(GpuScalarKind::Bool),
            },
            GpuExpr::Field { base, field } => {
                if let GpuValueType::Struct(sname) = self.infer_expr_type(base) {
                    self.lookup_field(sname, field)
                } else {
                    GpuValueType::Unknown
                }
            }
            GpuExpr::Index { base, .. } => {
                if let GpuValueType::Array(elem_kind, _) = self.infer_expr_type(base) {
                    GpuValueType::Scalar(elem_kind)
                } else {
                    GpuValueType::Unknown
                }
            }
            GpuExpr::Binary { op, lhs, rhs } => {
                if op.is_comparison_or_logical() {
                    return GpuValueType::Scalar(GpuScalarKind::Bool);
                }
                let lt = self.infer_expr_type(lhs);
                let rt = self.infer_expr_type(rhs);
                if matches!(lt, GpuValueType::Scalar(GpuScalarKind::U32))
                    || matches!(rt, GpuValueType::Scalar(GpuScalarKind::U32))
                {
                    return GpuValueType::Scalar(GpuScalarKind::U32);
                }
                if !matches!(lt, GpuValueType::Unknown) {
                    lt
                } else {
                    rt
                }
            }
            GpuExpr::Unary { op, expr } => match op {
                GpuUnaryOp::Not => GpuValueType::Scalar(GpuScalarKind::Bool),
                GpuUnaryOp::Neg => self.infer_expr_type(expr),
            },
            GpuExpr::Paren(inner) => self.infer_expr_type(inner),
            GpuExpr::Cast { ty, .. } => GpuValueType::Scalar(*ty),
            GpuExpr::MathCall {
                receiver,
                intrinsic,
                ..
            } => match intrinsic {
                GpuMathIntrinsic::Abs
                | GpuMathIntrinsic::Min
                | GpuMathIntrinsic::Max
                | GpuMathIntrinsic::Clamp => self.infer_expr_type(receiver),
                _ => GpuValueType::Scalar(GpuScalarKind::F32),
            },
            GpuExpr::HelperCall { helper, .. } => GpuValueType::Scalar(helper.return_type),
        }
    }

    fn is_strictly_unsigned_expr(&self, expr: &GpuExpr) -> bool {
        match expr {
            GpuExpr::Lit(GpuLit::UntypedInt(_)) => false,
            _ => matches!(
                self.infer_expr_type(expr),
                GpuValueType::Scalar(GpuScalarKind::U32)
            ),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IrBackendTarget {
    Msl,
    Wgsl,
}

struct IrCodeEmitter {
    target: IrBackendTarget,
}

impl IrCodeEmitter {
    fn new(target: IrBackendTarget) -> Self {
        Self { target }
    }

    fn emit_block(&self, block: &GpuBlock, env: &mut ShaderTypeEnv<'_>) -> String {
        env.push_scope();
        let mut out = String::new();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            if idx > 0 {
                out.push(' ');
            }
            out.push_str(&self.emit_stmt(stmt, env));
        }
        env.pop_scope();
        out
    }

    fn emit_stmt(&self, stmt: &GpuStmt, env: &mut ShaderTypeEnv<'_>) -> String {
        match stmt {
            GpuStmt::Let {
                name,
                mutable,
                explicit_ty,
                init,
            } => {
                let coerce_u32 = *explicit_ty == Some(GpuScalarKind::U32);
                let rhs = self.emit_expr(init, env, coerce_u32);
                let inferred = if let Some(kind) = explicit_ty {
                    GpuValueType::Scalar(*kind)
                } else {
                    env.infer_expr_type(init)
                };
                env.define_var(name, inferred);
                match (self.target, *mutable, explicit_ty) {
                    (IrBackendTarget::Msl, true, Some(ty)) => {
                        format!("{} {name} = {rhs};", ty.msl_type())
                    }
                    (IrBackendTarget::Msl, false, Some(ty)) => {
                        format!("const {} {name} = {rhs};", ty.msl_type())
                    }
                    (IrBackendTarget::Msl, true, None) => format!("auto {name} = {rhs};"),
                    (IrBackendTarget::Msl, false, None) => format!("const auto {name} = {rhs};"),
                    (IrBackendTarget::Wgsl, true, Some(ty)) => {
                        format!("var {name}: {} = {rhs};", ty.wgsl_type())
                    }
                    (IrBackendTarget::Wgsl, false, Some(ty)) => {
                        format!("let {name}: {} = {rhs};", ty.wgsl_type())
                    }
                    (IrBackendTarget::Wgsl, true, None) => format!("var {name} = {rhs};"),
                    (IrBackendTarget::Wgsl, false, None) => format!("let {name} = {rhs};"),
                }
            }
            GpuStmt::Assign { op, target, value } => {
                let lhs = self.emit_lvalue(target, env);
                let rhs = self.emit_expr(value, env, false);
                format!("{lhs} {} {rhs};", op.as_str())
            }
            GpuStmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                let cond_str = self.emit_expr(cond, env, false);
                let then_str = self.emit_block(then_block, env);
                let mut s = format!("if ({cond_str}) {{ {then_str} }}");
                if let Some(eb) = else_branch {
                    match eb {
                        GpuElseBranch::ElseIf(elif_stmt) => {
                            let elif_str = self.emit_stmt(elif_stmt, env);
                            s.push_str(&format!(" else {elif_str}"));
                        }
                        GpuElseBranch::ElseBlock(eblock) => {
                            let else_str = self.emit_block(eblock, env);
                            s.push_str(&format!(" else {{ {else_str} }}"));
                        }
                    }
                }
                s
            }
            GpuStmt::While { cond, body } => {
                let cond_str = self.emit_expr(cond, env, false);
                let body_str = self.emit_block(body, env);
                format!("while ({cond_str}) {{ {body_str} }}")
            }
            GpuStmt::ForRange {
                var,
                start,
                end,
                inclusive,
                body,
            } => {
                let is_unsigned =
                    env.is_strictly_unsigned_expr(start) || env.is_strictly_unsigned_expr(end);
                let start_str = self.emit_expr(start, env, is_unsigned);
                let end_str = self.emit_expr(end, env, is_unsigned);
                let step_str = if is_unsigned { "1u" } else { "1" };
                let cmp_str = if *inclusive { "<=" } else { "<" };

                env.push_scope();
                env.define_var(
                    var,
                    GpuValueType::Scalar(if is_unsigned {
                        GpuScalarKind::U32
                    } else {
                        GpuScalarKind::I32
                    }),
                );
                let body_str = self.emit_block(body, env);
                env.pop_scope();

                match self.target {
                    IrBackendTarget::Msl => format!(
                        "for (auto {var} = {start_str}; {var} {cmp_str} {end_str}; {var} += {step_str}) {{ {body_str} }}"
                    ),
                    IrBackendTarget::Wgsl => format!(
                        "for (var {var} = {start_str}; {var} {cmp_str} {end_str}; {var} += {step_str}) {{ {body_str} }}"
                    ),
                }
            }
            GpuStmt::Block(inner) => {
                let inner_str = self.emit_block(inner, env);
                format!("{{ {inner_str} }}")
            }
            GpuStmt::Return(opt_val) => {
                if let Some(val) = opt_val {
                    let val_str = self.emit_expr(val, env, false);
                    format!("return {val_str};")
                } else {
                    match self.target {
                        IrBackendTarget::Msl => "return;".to_string(),
                        IrBackendTarget::Wgsl => {
                            if env.mut_params.is_empty() {
                                "return;".to_string()
                            } else {
                                let mut wb = String::new();
                                for p in env.mut_params {
                                    wb.push_str(&format!("sumeru_buf_{p}[gid] = {p}; "));
                                }
                                wb.push_str("return;");
                                wb
                            }
                        }
                    }
                }
            }
            GpuStmt::Break => "break;".to_string(),
            GpuStmt::Continue => "continue;".to_string(),
        }
    }

    fn emit_lvalue(&self, lv: &GpuLValue, env: &mut ShaderTypeEnv<'_>) -> String {
        match lv {
            GpuLValue::Var(name) => (*name).to_string(),
            GpuLValue::Field { base, field } => {
                let b = self.emit_lvalue(base, env);
                format!("{b}.{field}")
            }
            GpuLValue::Index { base, index } => {
                let b = self.emit_lvalue(base, env);
                let idx = self.emit_expr(index, env, false);
                format!("{b}[{idx}]")
            }
        }
    }

    fn emit_expr(
        &self,
        expr: &GpuExpr,
        env: &mut ShaderTypeEnv<'_>,
        coerce_unsigned: bool,
    ) -> String {
        match expr {
            GpuExpr::Var(name) => (*name).to_string(),
            GpuExpr::Lit(lit) => lit.format_shader(coerce_unsigned),
            GpuExpr::Field { base, field } => {
                let b = self.emit_expr(base, env, false);
                format!("{b}.{field}")
            }
            GpuExpr::Index { base, index } => {
                let b = self.emit_expr(base, env, false);
                let idx = self.emit_expr(index, env, false);
                format!("{b}[{idx}]")
            }
            GpuExpr::Binary { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, env, false);
                let r = self.emit_expr(rhs, env, false);
                format!("{l} {} {r}", op.as_str())
            }
            GpuExpr::Unary { op, expr } => {
                let inner = self.emit_expr(expr, env, false);
                match op {
                    GpuUnaryOp::Neg => format!("-({inner})"),
                    GpuUnaryOp::Not => format!("!({inner})"),
                }
            }
            GpuExpr::Paren(inner) => {
                let s = self.emit_expr(inner, env, coerce_unsigned);
                format!("({s})")
            }
            GpuExpr::Cast { expr, ty } => {
                let inner = self.emit_expr(expr, env, false);
                match self.target {
                    IrBackendTarget::Msl => format!("{}({inner})", ty.msl_type()),
                    IrBackendTarget::Wgsl => format!("{}({inner})", ty.wgsl_type()),
                }
            }
            GpuExpr::MathCall {
                receiver,
                intrinsic,
                args,
            } => {
                let recv = self.emit_expr(receiver, env, false);
                let emitted_args: Vec<String> =
                    args.iter().map(|a| self.emit_expr(a, env, false)).collect();
                match intrinsic {
                    GpuMathIntrinsic::Abs => format!("abs({recv})"),
                    GpuMathIntrinsic::Sqrt => format!("sqrt({recv})"),
                    GpuMathIntrinsic::Sin => format!("sin({recv})"),
                    GpuMathIntrinsic::Cos => format!("cos({recv})"),
                    GpuMathIntrinsic::Tan => format!("tan({recv})"),
                    GpuMathIntrinsic::Floor => format!("floor({recv})"),
                    GpuMathIntrinsic::Ceil => format!("ceil({recv})"),
                    GpuMathIntrinsic::Round => format!("round({recv})"),
                    GpuMathIntrinsic::Fract => format!("fract({recv})"),
                    GpuMathIntrinsic::Exp => format!("exp({recv})"),
                    GpuMathIntrinsic::Ln => format!("log({recv})"),
                    GpuMathIntrinsic::Min => format!("min({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Max => format!("max({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Atan2 => format!("atan2({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Powf => format!("pow({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Clamp => {
                        format!("clamp({recv}, {}, {})", emitted_args[0], emitted_args[1])
                    }
                    GpuMathIntrinsic::Fma => {
                        format!("fma({recv}, {}, {})", emitted_args[0], emitted_args[1])
                    }
                }
            }
            GpuExpr::HelperCall { helper, args } => {
                let emitted_args: Vec<String> =
                    args.iter().map(|a| self.emit_expr(a, env, false)).collect();
                format!("{}({})", helper.name, emitted_args.join(", "))
            }
        }
    }

    fn emit_helper_fn(&self, helper: &GpuHelperFnDescriptor) -> String {
        let mut env = ShaderTypeEnv::for_helper(&helper.params);
        match self.target {
            IrBackendTarget::Msl => {
                let params_str: Vec<String> = helper
                    .params
                    .iter()
                    .map(|&(name, _, kind)| format!("{} {name}", kind.msl_type()))
                    .collect();
                let body_str = self.emit_block(&helper.body, &mut env);
                format!(
                    "inline {} {}({}) {{\n    {}\n}}\n",
                    helper.return_type.msl_type(),
                    helper.name,
                    params_str.join(", "),
                    body_str
                )
            }
            IrBackendTarget::Wgsl => {
                let mut prelude = String::new();
                let params_str: Vec<String> = helper
                    .params
                    .iter()
                    .map(|&(name, is_mut, kind)| {
                        if is_mut {
                            prelude.push_str(&format!(
                                "var {name}: {} = __sumeru_arg_{name}; ",
                                kind.wgsl_type()
                            ));
                            format!("__sumeru_arg_{name}: {}", kind.wgsl_type())
                        } else {
                            format!("{name}: {}", kind.wgsl_type())
                        }
                    })
                    .collect();
                let body_str = self.emit_block(&helper.body, &mut env);
                format!(
                    "fn {}({}) -> {} {{\n    {}{}\n}}\n",
                    helper.name,
                    params_str.join(", "),
                    helper.return_type.wgsl_type(),
                    prelude,
                    body_str
                )
            }
        }
    }
}

/// 递归按后序（被调用者在前，调用者在后）收集 `GpuBlock` 中引用的全部 `#[gpu]` 纯计算辅助函数。
fn collect_helpers_topological(
    block: &GpuBlock,
    out: &mut Vec<&'static GpuHelperFnDescriptor>,
) {
    for stmt in &block.stmts {
        collect_helpers_in_stmt(stmt, out);
    }
}

fn collect_helpers_in_stmt(stmt: &GpuStmt, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match stmt {
        GpuStmt::Let { init, .. } => collect_helpers_in_expr(init, out),
        GpuStmt::Assign { target, value, .. } => {
            collect_helpers_in_lvalue(target, out);
            collect_helpers_in_expr(value, out);
        }
        GpuStmt::If {
            cond,
            then_block,
            else_branch,
        } => {
            collect_helpers_in_expr(cond, out);
            collect_helpers_topological(then_block, out);
            if let Some(eb) = else_branch {
                match eb {
                    GpuElseBranch::ElseIf(s) => collect_helpers_in_stmt(s, out),
                    GpuElseBranch::ElseBlock(b) => collect_helpers_topological(b, out),
                }
            }
        }
        GpuStmt::While { cond, body } => {
            collect_helpers_in_expr(cond, out);
            collect_helpers_topological(body, out);
        }
        GpuStmt::ForRange {
            start, end, body, ..
        } => {
            collect_helpers_in_expr(start, out);
            collect_helpers_in_expr(end, out);
            collect_helpers_topological(body, out);
        }
        GpuStmt::Block(b) => collect_helpers_topological(b, out),
        GpuStmt::Return(Some(e)) => collect_helpers_in_expr(e, out),
        GpuStmt::Return(None) | GpuStmt::Break | GpuStmt::Continue => {}
    }
}

fn collect_helpers_in_lvalue(lv: &GpuLValue, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match lv {
        GpuLValue::Var(_) => {}
        GpuLValue::Field { base, .. } => collect_helpers_in_lvalue(base, out),
        GpuLValue::Index { base, index } => {
            collect_helpers_in_lvalue(base, out);
            collect_helpers_in_expr(index, out);
        }
    }
}

fn collect_helpers_in_expr(expr: &GpuExpr, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match expr {
        GpuExpr::Var(_) | GpuExpr::Lit(_) => {}
        GpuExpr::Field { base, .. }
        | GpuExpr::Unary { expr: base, .. }
        | GpuExpr::Paren(base)
        | GpuExpr::Cast { expr: base, .. } => collect_helpers_in_expr(base, out),
        GpuExpr::Index { base, index } => {
            collect_helpers_in_expr(base, out);
            collect_helpers_in_expr(index, out);
        }
        GpuExpr::Binary { lhs, rhs, .. } => {
            collect_helpers_in_expr(lhs, out);
            collect_helpers_in_expr(rhs, out);
        }
        GpuExpr::MathCall { receiver, args, .. } => {
            collect_helpers_in_expr(receiver, out);
            for a in args {
                collect_helpers_in_expr(a, out);
            }
        }
        GpuExpr::HelperCall { helper, args } => {
            for a in args {
                collect_helpers_in_expr(a, out);
            }
            if !out.iter().any(|h| std::ptr::eq(*h, *helper)) {
                collect_helpers_topological(&helper.body, out);
                if !out.iter().any(|h| h.name == helper.name) {
                    out.push(helper);
                }
            }
        }
    }
}

/// 从强类型 [`GpuBlock`] IR 与组件 Schema 元数据一次性编译生成完整的 MSL 2.0 与 WGSL 内核源码。
pub fn compile_kernel_ir_sources(
    kernel_name: &str,
    components: &[GpuComponentSchema],
    msl_struct_defs: &[String],
    wgsl_struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    ir_block: &GpuBlock,
) -> (String, String) {
    let mut helpers = Vec::new();
    collect_helpers_topological(ir_block, &mut helpers);

    let mut_params: Vec<&'static str> = components
        .iter()
        .filter(|c| c.mutable)
        .map(|c| c.param_name)
        .collect();

    let msl_emitter = IrCodeEmitter::new(IrBackendTarget::Msl);
    let wgsl_emitter = IrCodeEmitter::new(IrBackendTarget::Wgsl);

    let mut msl_helpers = String::new();
    let mut wgsl_helpers = String::new();
    for h in &helpers {
        msl_helpers.push_str(&msl_emitter.emit_helper_fn(h));
        msl_helpers.push('\n');
        wgsl_helpers.push_str(&wgsl_emitter.emit_helper_fn(h));
        wgsl_helpers.push('\n');
    }

    let mut msl_env = ShaderTypeEnv::for_kernel(components, &mut_params);
    let mut wgsl_env = ShaderTypeEnv::for_kernel(components, &mut_params);

    let msl_body = msl_emitter.emit_block(ir_block, &mut msl_env);
    let wgsl_body = wgsl_emitter.emit_block(ir_block, &mut wgsl_env);

    let mut msl_source = super::kernel::build_msl_kernel_source_with_helpers(
        kernel_name,
        msl_struct_defs,
        bindings,
        &msl_helpers,
        &msl_body,
    );
    let mut wgsl_source = super::kernel::build_wgsl_kernel_source_with_helpers(
        kernel_name,
        wgsl_struct_defs,
        bindings,
        &wgsl_helpers,
        &wgsl_body,
    );
    let _ = (&mut msl_source, &mut wgsl_source);
    (msl_source, wgsl_source)
}
