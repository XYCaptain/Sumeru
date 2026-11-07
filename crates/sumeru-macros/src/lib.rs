//! Procedural macros for Sumeru 16KB page-aligned GPU compute (`#[gpu]`, `#[gpu_component]`, `#[gpu_kernel]`).
//!
//! Uses `syn` AST parsing and strict shader subset validation to translate Rust kernel bodies
//! into Apple Metal Shading Language (MSL) and WebGPU Shading Language (WGSL) at compile time.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{
    BinOp, Block, Data, DeriveInput, Error, Expr, Fields, FnArg, GenericParam, Ident, Item,
    ItemFn, ItemStruct, Lit, Member, Pat, RangeLimits, ReturnType, Stmt, Type, UnOp,
};

/// Unified GPU attribute macro for Sumeru ECS.
///
/// - When placed on a `struct`, expands to a `#[repr(C)]` Pod/Zeroable GPU component with
///   automatic `GpuComponent` and `GpuShaderStruct` (MSL + WGSL) implementations.
/// - When placed on a `fn`, validates the Rust AST against the GPU compute subset and expands to a
///   dual-target operator that can be:
///   1. Called directly in Rust or inside `.for_each(|(a, b)| ...)` / `.par_for_each(|(a, b)| ...)`,
///   2. Passed directly to `.run(op)` or `.par_run(op)` / `.par().run(op)`,
///   3. Dispatched to GPU via `.gpu().for_each(op)?` or `.gpu().run(op)?`.
#[proc_macro_attribute]
pub fn gpu(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_gpu_item(attr.into(), item.into()).into()
}

/// Attribute macro for GPU-compatible `#[repr(C)]` ECS components.
#[proc_macro_attribute]
pub fn gpu_component(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_gpu_component_impl(attr.into(), item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Attribute macro for multi-platform GPU compute kernels (MSL + WGSL + CPU).
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: TokenStream2 = attr.into();
    let has_custom_type_name = !attr2.is_empty();
    expand_gpu_fn_impl(attr2, item.into(), !has_custom_type_name)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro for `GpuComponent`.
#[proc_macro_derive(GpuComponent)]
pub fn derive_gpu_component(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);
    match derive_gpu_component_impl(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Derive macro for `GpuShaderStruct` (generates MSL & WGSL struct definitions).
#[proc_macro_derive(GpuShaderStruct)]
pub fn derive_gpu_shader_struct(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);
    match derive_gpu_shader_struct_impl(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_gpu_item(attr: TokenStream2, item: TokenStream2) -> TokenStream2 {
    match syn::parse2::<Item>(item.clone()) {
        Ok(Item::Struct(_)) => {
            expand_gpu_component_impl(attr, item).unwrap_or_else(|e| e.to_compile_error())
        }
        Ok(Item::Fn(_)) => {
            expand_gpu_fn_impl(attr, item, true).unwrap_or_else(|e| e.to_compile_error())
        }
        Ok(other) => Error::new_spanned(
            other,
            "`#[gpu]` can only be applied to a `struct` or `fn` item",
        )
        .to_compile_error(),
        Err(e) => e.to_compile_error(),
    }
}

fn derive_gpu_component_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &input.generics,
            "`GpuComponent` cannot have generic parameters",
        ));
    }
    let Data::Struct(_) = &input.data else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuComponent)]` can only be applied to a struct",
        ));
    };
    let name = &input.ident;
    Ok(quote! {
        unsafe impl ::sumeru::GpuComponent for #name {}
    })
}

fn derive_gpu_shader_struct_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &input.generics,
            "`GpuShaderStruct` cannot have generic parameters",
        ));
    }
    let Data::Struct(data_struct) = &input.data else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` can only be applied to a named struct",
        ));
    };
    let Fields::Named(fields_named) = &data_struct.fields else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` requires named struct fields `{ ... }`",
        ));
    };
    if fields_named.named.is_empty() {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` struct must have at least one field",
        ));
    }
    let name = &input.ident;
    let name_str = name.to_string();
    let field_entries = fields_named.named.iter().map(|f| {
        let fname = f.ident.as_ref().unwrap().to_string();
        let fty = &f.ty;
        quote! {
            ::sumeru::GpuStructField {
                name: #fname,
                field_type: <#fty as ::sumeru::GpuScalarType>::FIELD_TYPE,
            }
        }
    });
    Ok(quote! {
        impl ::sumeru::GpuShaderStruct for #name {
            const STRUCT_NAME: &'static str = #name_str;
            const FIELDS: &'static [::sumeru::GpuStructField] = &[
                #(#field_entries),*
            ];
        }
    })
}

fn expand_gpu_component_impl(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    if !attr.is_empty() {
        return Err(Error::new_spanned(
            attr,
            "`#[gpu]` on a struct does not accept attribute arguments",
        ));
    }
    let item_struct: ItemStruct = syn::parse2(item)?;
    if !item_struct.generics.params.is_empty() || item_struct.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &item_struct.generics,
            "`#[gpu]` component structs cannot have generic parameters or where clauses",
        ));
    }
    let Fields::Named(fields_named) = &item_struct.fields else {
        return Err(Error::new_spanned(
            &item_struct,
            "`#[gpu]` on a type requires a named struct with `{ ... }` fields",
        ));
    };
    if fields_named.named.is_empty() {
        return Err(Error::new_spanned(
            &item_struct,
            "`#[gpu]` component structs cannot be zero-sized (must have at least one field)",
        ));
    }

    let attrs: Vec<_> = item_struct
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("repr"))
        .collect();
    let vis = &item_struct.vis;
    let name = &item_struct.ident;
    let fields = fields_named.named.iter().map(|f| {
        let fattrs = &f.attrs;
        let fvis = &f.vis;
        let fident = f.ident.as_ref().unwrap();
        let fty = &f.ty;
        quote! {
            #(#fattrs)*
            #fvis #fident : #fty
        }
    });

    Ok(quote! {
        ::sumeru::sumeru_gpu_struct! {
            #(#attrs)*
            #vis struct #name {
                #(#fields),*
            }
        }
    })
}

struct ParsedKernelParam {
    ident: Ident,
    is_mut: bool,
    elem_ty: Type,
}

fn expand_gpu_fn_impl(
    attr: TokenStream2,
    item: TokenStream2,
    same_name_callable: bool,
) -> syn::Result<TokenStream2> {
    let item_fn: ItemFn = syn::parse2(item)?;
    let sig = &item_fn.sig;

    if let Some(async_kw) = &sig.asyncness {
        return Err(Error::new_spanned(
            async_kw,
            "`#[gpu]` kernel functions cannot be `async`",
        ));
    }
    if let Some(const_kw) = &sig.constness {
        return Err(Error::new_spanned(
            const_kw,
            "`#[gpu]` kernel functions cannot be `const`",
        ));
    }
    if let Some(unsafe_kw) = &sig.unsafety {
        return Err(Error::new_spanned(
            unsafe_kw,
            "`#[gpu]` kernel functions cannot be `unsafe`",
        ));
    }
    if let Some(abi) = &sig.abi {
        return Err(Error::new_spanned(
            abi,
            "`#[gpu]` kernel functions cannot have an `extern` ABI",
        ));
    }
    if let Some(variadic) = &sig.variadic {
        return Err(Error::new_spanned(
            variadic,
            "`#[gpu]` kernel functions cannot be variadic",
        ));
    }
    for param in &sig.generics.params {
        match param {
            GenericParam::Lifetime(_) => {}
            GenericParam::Type(tp) => {
                return Err(Error::new_spanned(
                    tp,
                    "`#[gpu]` kernel functions cannot have generic type parameters",
                ));
            }
            GenericParam::Const(cp) => {
                return Err(Error::new_spanned(
                    cp,
                    "`#[gpu]` kernel functions cannot have const generic parameters",
                ));
            }
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        return Err(Error::new_spanned(
            where_clause,
            "`#[gpu]` kernel functions cannot have a `where` clause",
        ));
    }
    if let ReturnType::Type(_, ret_ty) = &sig.output {
        return Err(Error::new_spanned(
            ret_ty,
            "`#[gpu]` kernel functions must not have a return type",
        ));
    }

    let mut params = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(Error::new_spanned(
                arg,
                "`#[gpu]` kernel functions cannot take `self`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(Error::new_spanned(
                &pat_type.pat,
                "`#[gpu]` parameter pattern must be a simple identifier",
            ));
        };
        let pname = pat_ident.ident.to_string();
        if pname.starts_with("__") || pname.starts_with("sumeru_") {
            return Err(Error::new_spanned(
                &pat_ident.ident,
                "`#[gpu]` parameter name cannot start with reserved prefix `__` or `sumeru_`",
            ));
        }
        let Type::Reference(ty_ref) = &*pat_type.ty else {
            return Err(Error::new_spanned(
                &pat_type.ty,
                format!("parameter `{pname}` in `#[gpu]` kernel must be `&T` or `&mut T`"),
            ));
        };
        if matches!(&*ty_ref.elem, Type::Reference(_)) {
            return Err(Error::new_spanned(
                &ty_ref.elem,
                format!("parameter `{pname}` cannot be a nested reference"),
            ));
        }
        params.push(ParsedKernelParam {
            ident: pat_ident.ident.clone(),
            is_mut: ty_ref.mutability.is_some(),
            elem_ty: (*ty_ref.elem).clone(),
        });
    }

    if params.is_empty() || params.len() > 4 {
        return Err(Error::new_spanned(
            &sig.inputs,
            "`#[gpu]` kernel functions must have between 1 and 4 component reference parameters",
        ));
    }

    let mut_params: Vec<String> = params
        .iter()
        .filter(|p| p.is_mut)
        .map(|p| p.ident.to_string())
        .collect();

    let msl_body = ShaderTranslator::new(ShaderTarget::Msl, &mut_params)
        .translate_block(&item_fn.block)?;
    let wgsl_body = ShaderTranslator::new(ShaderTarget::Wgsl, &mut_params)
        .translate_block(&item_fn.block)?;

    let attrs = &item_fn.attrs;
    let vis = &item_fn.vis;
    let fn_name = &sig.ident;
    let stmts = &item_fn.block.stmts;

    let param_tokens = params.iter().map(|p| {
        let ident = &p.ident;
        let mode = if p.is_mut {
            Ident::new("mut", ident.span())
        } else {
            Ident::new("ref", ident.span())
        };
        let ty = &p.elem_ty;
        quote! { #ident : [#mode] #ty }
    });

    if same_name_callable {
        if !attr.is_empty() {
            return Err(Error::new_spanned(
                attr,
                "`#[gpu]` on a function does not accept attribute arguments; use `#[gpu_kernel(CustomName)]` if needed",
            ));
        }
        Ok(quote! {
            ::sumeru::sumeru_gpu_fn! {
                @translated
                #(#attrs)*
                #vis fn #fn_name ( #(#param_tokens),* )
                msl = #msl_body,
                wgsl = #wgsl_body,
                body = { #(#stmts)* }
            }
        })
    } else {
        let custom_ident: Ident = syn::parse2(attr)?;
        Ok(quote! {
            ::sumeru::sumeru_gpu_kernel! {
                @translated
                #(#attrs)*
                #vis kernel #custom_ident fn #fn_name ( #(#param_tokens),* )
                msl = #msl_body,
                wgsl = #wgsl_body,
                body = { #(#stmts)* }
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShaderTarget {
    Msl,
    Wgsl,
}

struct ShaderTranslator<'a> {
    target: ShaderTarget,
    mut_params: &'a [String],
}

impl<'a> ShaderTranslator<'a> {
    fn new(target: ShaderTarget, mut_params: &'a [String]) -> Self {
        Self { target, mut_params }
    }

    fn translate_block(&self, block: &Block) -> syn::Result<String> {
        let mut out = String::new();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            if idx > 0 {
                out.push(' ');
            }
            out.push_str(&self.translate_stmt(stmt)?);
        }
        Ok(out)
    }

    fn translate_stmt(&self, stmt: &Stmt) -> syn::Result<String> {
        match stmt {
            Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return Err(Error::new_spanned(
                        local,
                        "`#[gpu]` local variables must be initialized at declaration",
                    ));
                };
                if init.diverge.is_some() {
                    return Err(Error::new_spanned(
                        local,
                        "`let ... else` is not supported in `#[gpu]` kernels",
                    ));
                }
                let (ident, is_mut, explicit_ty) = match &local.pat {
                    Pat::Ident(pi) => (&pi.ident, pi.mutability.is_some(), None),
                    Pat::Type(pt) => {
                        let Pat::Ident(pi) = &*pt.pat else {
                            return Err(Error::new_spanned(
                                &pt.pat,
                                "unsupported pattern in `#[gpu]` local variable declaration",
                            ));
                        };
                        let ty_str = self.translate_scalar_type(&pt.ty)?;
                        (&pi.ident, pi.mutability.is_some(), Some(ty_str))
                    }
                    other => {
                        return Err(Error::new_spanned(
                            other,
                            "destructuring patterns in `let` are not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                let vname = ident.to_string();
                if vname.starts_with("__") || vname.starts_with("sumeru_") {
                    return Err(Error::new_spanned(
                        ident,
                        "local variable name in `#[gpu]` kernel cannot start with `__` or `sumeru_`",
                    ));
                }
                let rhs = self.translate_expr(&init.expr)?;
                match (self.target, is_mut, explicit_ty) {
                    (ShaderTarget::Msl, true, Some(ty)) => Ok(format!("{ty} {vname} = {rhs};")),
                    (ShaderTarget::Msl, false, Some(ty)) => {
                        Ok(format!("const {ty} {vname} = {rhs};"))
                    }
                    (ShaderTarget::Msl, true, None) => Ok(format!("auto {vname} = {rhs};")),
                    (ShaderTarget::Msl, false, None) => Ok(format!("const auto {vname} = {rhs};")),
                    (ShaderTarget::Wgsl, true, Some(ty)) => {
                        Ok(format!("var {vname}: {ty} = {rhs};"))
                    }
                    (ShaderTarget::Wgsl, false, Some(ty)) => {
                        Ok(format!("let {vname}: {ty} = {rhs};"))
                    }
                    (ShaderTarget::Wgsl, true, None) => Ok(format!("var {vname} = {rhs};")),
                    (ShaderTarget::Wgsl, false, None) => Ok(format!("let {vname} = {rhs};")),
                }
            }
            Stmt::Item(item) => Err(Error::new_spanned(
                item,
                "nested item declarations are not supported in `#[gpu]` kernels",
            )),
            Stmt::Macro(mac) => Err(Error::new_spanned(
                mac,
                "macros are not supported inside `#[gpu]` kernel bodies",
            )),
            Stmt::Expr(expr, semi) => self.translate_stmt_expr(expr, semi.is_some()),
        }
    }

    fn translate_stmt_expr(&self, expr: &Expr, has_semi: bool) -> syn::Result<String> {
        match expr {
            Expr::Assign(assign) => {
                let lhs = self.translate_lvalue(&assign.left)?;
                let rhs = self.translate_expr(&assign.right)?;
                Ok(format!("{lhs} = {rhs};"))
            }
            Expr::Binary(bin) if is_compound_assign(&bin.op) => {
                let lhs = self.translate_lvalue(&bin.left)?;
                let op = translate_binop(&bin.op)?;
                let rhs = self.translate_expr(&bin.right)?;
                Ok(format!("{lhs} {op} {rhs};"))
            }
            Expr::If(expr_if) => {
                let cond = self.translate_expr(&expr_if.cond)?;
                let then_body = self.translate_block(&expr_if.then_branch)?;
                let mut s = format!("if ({cond}) {{ {then_body} }}");
                if let Some((_, else_expr)) = &expr_if.else_branch {
                    match &**else_expr {
                        Expr::If(_) => {
                            let else_if = self.translate_stmt_expr(else_expr, true)?;
                            s.push_str(&format!(" else {else_if}"));
                        }
                        Expr::Block(eb) => {
                            let else_body = self.translate_block(&eb.block)?;
                            s.push_str(&format!(" else {{ {else_body} }}"));
                        }
                        other => {
                            return Err(Error::new_spanned(
                                other,
                                "unsupported `else` branch in `#[gpu]` kernel",
                            ));
                        }
                    }
                }
                Ok(s)
            }
            Expr::While(expr_while) => {
                let cond = self.translate_expr(&expr_while.cond)?;
                let body = self.translate_block(&expr_while.body)?;
                Ok(format!("while ({cond}) {{ {body} }}"))
            }
            Expr::ForLoop(expr_for) => {
                let Pat::Ident(pat_ident) = &*expr_for.pat else {
                    return Err(Error::new_spanned(
                        &expr_for.pat,
                        "`for` loop pattern in `#[gpu]` kernel must be a simple identifier",
                    ));
                };
                let var_name = pat_ident.ident.to_string();
                let Expr::Range(range) = &*expr_for.expr else {
                    return Err(Error::new_spanned(
                        &expr_for.expr,
                        "`for` loop in `#[gpu]` kernel only supports `start..end` or `start..=end` ranges",
                    ));
                };
                let (Some(start_expr), Some(end_expr)) = (&range.start, &range.end) else {
                    return Err(Error::new_spanned(
                        range,
                        "`for` loop range in `#[gpu]` kernel must specify both start and end bounds",
                    ));
                };
                let start = self.translate_expr(start_expr)?;
                let end = self.translate_expr(end_expr)?;
                let cmp = match range.limits {
                    RangeLimits::HalfOpen(_) => "<",
                    RangeLimits::Closed(_) => "<=",
                };
                let body = self.translate_block(&expr_for.body)?;
                match self.target {
                    ShaderTarget::Msl => Ok(format!(
                        "for (auto {var_name} = {start}; {var_name} {cmp} {end}; {var_name} += 1) {{ {body} }}"
                    )),
                    ShaderTarget::Wgsl => Ok(format!(
                        "for (var {var_name} = {start}; {var_name} {cmp} {end}; {var_name} += 1) {{ {body} }}"
                    )),
                }
            }
            Expr::Block(eb) => {
                if eb.label.is_some() {
                    return Err(Error::new_spanned(
                        &eb.label,
                        "labeled blocks are not supported in `#[gpu]` kernels",
                    ));
                }
                let inner = self.translate_block(&eb.block)?;
                Ok(format!("{{ {inner} }}"))
            }
            Expr::Return(ret) => {
                if let Some(val) = &ret.expr {
                    return Err(Error::new_spanned(
                        val,
                        "`#[gpu]` kernel cannot return a value",
                    ));
                }
                match self.target {
                    ShaderTarget::Msl => Ok("return;".into()),
                    ShaderTarget::Wgsl => {
                        if self.mut_params.is_empty() {
                            Ok("return;".into())
                        } else {
                            let mut wb = String::new();
                            for p in self.mut_params {
                                wb.push_str(&format!("sumeru_buf_{p}[gid] = {p}; "));
                            }
                            wb.push_str("return;");
                            Ok(wb)
                        }
                    }
                }
            }
            Expr::Break(brk) => {
                if brk.label.is_some() || brk.expr.is_some() {
                    return Err(Error::new_spanned(
                        brk,
                        "labeled or value `break` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok("break;".into())
            }
            Expr::Continue(cont) => {
                if cont.label.is_some() {
                    return Err(Error::new_spanned(
                        cont,
                        "labeled `continue` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok("continue;".into())
            }
            other => {
                if !has_semi {
                    return Err(Error::new_spanned(
                        other,
                        "`#[gpu]` kernel cannot end with an implicit return expression; add a semicolon or use an assignment",
                    ));
                }
                Err(Error::new_spanned(
                    other,
                    "unsupported statement in `#[gpu]` kernel; only assignments, local `let`, `if`/`else`, `while`, `for`, and `return;` are supported",
                ))
            }
        }
    }

    fn translate_lvalue(&self, expr: &Expr) -> syn::Result<String> {
        match expr {
            Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
                Ok(p.path.segments[0].ident.to_string())
            }
            Expr::Field(f) => {
                let base = self.translate_lvalue(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(format!("{base}.{member}"))
            }
            Expr::Index(idx) => {
                let base = self.translate_lvalue(&idx.expr)?;
                let index = self.translate_expr(&idx.index)?;
                Ok(format!("{base}[{index}]"))
            }
            Expr::Unary(un) if matches!(un.op, UnOp::Deref(_)) => self.translate_lvalue(&un.expr),
            Expr::Paren(p) => self.translate_lvalue(&p.expr),
            other => Err(Error::new_spanned(
                other,
                "invalid assignment target in `#[gpu]` kernel",
            )),
        }
    }

    fn translate_expr(&self, expr: &Expr) -> syn::Result<String> {
        match expr {
            Expr::Path(p) if p.qself.is_none() => {
                if p.path.segments.len() == 1 {
                    Ok(p.path.segments[0].ident.to_string())
                } else {
                    Err(Error::new_spanned(
                        p,
                        "multi-segment paths and external constants are not supported in `#[gpu]` kernels",
                    ))
                }
            }
            Expr::Field(f) => {
                let base = self.translate_expr(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(format!("{base}.{member}"))
            }
            Expr::Index(idx) => {
                let base = self.translate_expr(&idx.expr)?;
                let index = self.translate_expr(&idx.index)?;
                Ok(format!("{base}[{index}]"))
            }
            Expr::Binary(bin) => {
                if is_compound_assign(&bin.op) {
                    return Err(Error::new_spanned(
                        bin,
                        "compound assignment cannot be used as a sub-expression in `#[gpu]` kernels",
                    ));
                }
                let lhs = self.translate_expr(&bin.left)?;
                let op = translate_binop(&bin.op)?;
                let rhs = self.translate_expr(&bin.right)?;
                Ok(format!("{lhs} {op} {rhs}"))
            }
            Expr::Unary(un) => match un.op {
                UnOp::Neg(_) => {
                    let inner = self.translate_expr(&un.expr)?;
                    Ok(format!("-({inner})"))
                }
                UnOp::Not(_) => {
                    let inner = self.translate_expr(&un.expr)?;
                    Ok(format!("!({inner})"))
                }
                UnOp::Deref(_) => self.translate_expr(&un.expr),
                _ => Err(Error::new_spanned(
                    un,
                    "unsupported unary operator in `#[gpu]` kernel",
                )),
            },
            Expr::Paren(p) => {
                let inner = self.translate_expr(&p.expr)?;
                Ok(format!("({inner})"))
            }
            Expr::Cast(cast) => {
                let inner = self.translate_expr(&cast.expr)?;
                let ty = self.translate_scalar_type(&cast.ty)?;
                Ok(format!("{ty}({inner})"))
            }
            Expr::Lit(expr_lit) => self.translate_lit(&expr_lit.lit),
            Expr::MethodCall(mc) => {
                let receiver = self.translate_expr(&mc.receiver)?;
                let method = mc.method.to_string();
                let mut args = Vec::with_capacity(mc.args.len());
                for arg in &mc.args {
                    args.push(self.translate_expr(arg)?);
                }
                match (method.as_str(), args.len()) {
                    (
                        "abs" | "sqrt" | "sin" | "cos" | "tan" | "floor" | "ceil" | "round"
                        | "fract" | "exp",
                        0,
                    ) => Ok(format!("{method}({receiver})")),
                    ("ln", 0) => Ok(format!("log({receiver})")),
                    ("min" | "max" | "atan2", 1) => Ok(format!("{method}({receiver}, {})", args[0])),
                    ("powf", 1) => Ok(format!("pow({receiver}, {})", args[0])),
                    ("clamp", 2) => Ok(format!("clamp({receiver}, {}, {})", args[0], args[1])),
                    ("mul_add" | "fma", 2) => {
                        Ok(format!("fma({receiver}, {}, {})", args[0], args[1]))
                    }
                    _ => Err(Error::new_spanned(
                        &mc.method,
                        format!(
                            "method `{method}` (with {} arg(s)) is not supported in `#[gpu]` kernels; supported methods: abs, sqrt, min, max, clamp, sin, cos, tan, floor, ceil, round, fract, exp, ln, powf, atan2, mul_add",
                            args.len()
                        ),
                    )),
                }
            }
            other => Err(Error::new_spanned(
                other,
                "unsupported Rust expression in `#[gpu]` kernel; arbitrary function calls, closures, `match`, and references are not permitted in GPU shaders",
            )),
        }
    }

    fn translate_lit(&self, lit: &Lit) -> syn::Result<String> {
        match lit {
            Lit::Float(fl) => {
                let digits = fl.base10_digits().replace('_', "");
                if digits.contains('.') || digits.contains('e') || digits.contains('E') {
                    Ok(digits)
                } else {
                    Ok(format!("{digits}.0"))
                }
            }
            Lit::Int(il) => {
                let digits = il.base10_digits();
                let suffix = il.suffix();
                match suffix {
                    "" | "i32" => Ok(digits.to_string()),
                    "u32" | "usize" => Ok(format!("{digits}u")),
                    other => Err(Error::new_spanned(
                        il,
                        format!(
                            "integer literal suffix `{other}` is not supported in `#[gpu]` kernels; use `i32` or `u32`"
                        ),
                    )),
                }
            }
            Lit::Bool(b) => Ok(if b.value { "true".into() } else { "false".into() }),
            other => Err(Error::new_spanned(
                other,
                "string/byte/char literals are not supported in `#[gpu]` kernels",
            )),
        }
    }

    fn translate_scalar_type(&self, ty: &Type) -> syn::Result<&'static str> {
        let s = ty.to_token_stream().to_string();
        match (self.target, s.trim()) {
            (ShaderTarget::Msl, "f32") => Ok("float"),
            (ShaderTarget::Msl, "i32") => Ok("int"),
            (ShaderTarget::Msl, "u32") => Ok("uint"),
            (ShaderTarget::Wgsl, "f32") => Ok("f32"),
            (ShaderTarget::Wgsl, "i32") => Ok("i32"),
            (ShaderTarget::Wgsl, "u32") => Ok("u32"),
            _ => Err(Error::new_spanned(
                ty,
                format!(
                    "type `{s}` is not supported in `#[gpu]` scalar annotations/casts; use `f32`, `i32`, or `u32`"
                ),
            )),
        }
    }
}

fn is_compound_assign(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

fn translate_binop(op: &BinOp) -> syn::Result<&'static str> {
    match op {
        BinOp::Add(_) => Ok("+"),
        BinOp::Sub(_) => Ok("-"),
        BinOp::Mul(_) => Ok("*"),
        BinOp::Div(_) => Ok("/"),
        BinOp::Rem(_) => Ok("%"),
        BinOp::And(_) => Ok("&&"),
        BinOp::Or(_) => Ok("||"),
        BinOp::BitXor(_) => Ok("^"),
        BinOp::BitAnd(_) => Ok("&"),
        BinOp::BitOr(_) => Ok("|"),
        BinOp::Shl(_) => Ok("<<"),
        BinOp::Shr(_) => Ok(">>"),
        BinOp::Eq(_) => Ok("=="),
        BinOp::Lt(_) => Ok("<"),
        BinOp::Le(_) => Ok("<="),
        BinOp::Ne(_) => Ok("!="),
        BinOp::Ge(_) => Ok(">="),
        BinOp::Gt(_) => Ok(">"),
        BinOp::AddAssign(_) => Ok("+="),
        BinOp::SubAssign(_) => Ok("-="),
        BinOp::MulAssign(_) => Ok("*="),
        BinOp::DivAssign(_) => Ok("/="),
        BinOp::RemAssign(_) => Ok("%="),
        BinOp::BitXorAssign(_) => Ok("^="),
        BinOp::BitAndAssign(_) => Ok("&="),
        BinOp::BitOrAssign(_) => Ok("|="),
        BinOp::ShlAssign(_) => Ok("<<="),
        BinOp::ShrAssign(_) => Ok(">>="),
        other => Err(Error::new_spanned(
            other,
            "unsupported binary operator in `#[gpu]` kernel",
        )),
    }
}
