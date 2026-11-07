//! Procedural macros for Sumeru 16KB page-aligned GPU compute (`#[gpu]`, `#[gpu_component]`, `#[gpu_kernel]`).
//!
//! Lowers Rust `syn` AST into Sumeru's target-independent Typed Shader IR (`GpuBlock` / `GpuStmt` / `GpuExpr`)
//! with lexical scope tracking, compile-time `const` evaluation (`GpuConstValue`), and composable `#[gpu]`
//! pure helper functions (`GpuHelperFnDescriptor`).

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
/// - When placed on a `fn` with `&T` / `&mut T` component parameters, lowers the Rust AST into
///   a Typed Shader IR (`GpuBlock`) and expands to a three-tier operator (`run`, `par_run`, `gpu().for_each`).
/// - When placed on a pure scalar `fn(f32, ...) -> f32`, expands to a composable `#[gpu]` device helper
///   function that can be called both on CPU and from inside other `#[gpu]` kernels/helpers.
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

struct ParsedHelperParam {
    ident: Ident,
    is_mut: bool,
    ty: Type,
    kind_tokens: TokenStream2,
}

fn validate_fn_common_signature(sig: &syn::Signature) -> syn::Result<()> {
    if let Some(async_kw) = &sig.asyncness {
        return Err(Error::new_spanned(
            async_kw,
            "`#[gpu]` functions cannot be `async`",
        ));
    }
    if let Some(const_kw) = &sig.constness {
        return Err(Error::new_spanned(
            const_kw,
            "`#[gpu]` functions cannot be `const`",
        ));
    }
    if let Some(unsafe_kw) = &sig.unsafety {
        return Err(Error::new_spanned(
            unsafe_kw,
            "`#[gpu]` functions cannot be `unsafe`",
        ));
    }
    if let Some(abi) = &sig.abi {
        return Err(Error::new_spanned(
            abi,
            "`#[gpu]` functions cannot have an `extern` ABI",
        ));
    }
    if let Some(variadic) = &sig.variadic {
        return Err(Error::new_spanned(
            variadic,
            "`#[gpu]` functions cannot be variadic",
        ));
    }
    for param in &sig.generics.params {
        match param {
            GenericParam::Lifetime(_) => {}
            GenericParam::Type(tp) => {
                return Err(Error::new_spanned(
                    tp,
                    "`#[gpu]` functions cannot have generic type parameters",
                ));
            }
            GenericParam::Const(cp) => {
                return Err(Error::new_spanned(
                    cp,
                    "`#[gpu]` functions cannot have const generic parameters",
                ));
            }
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        return Err(Error::new_spanned(
            where_clause,
            "`#[gpu]` functions cannot have a `where` clause",
        ));
    }
    Ok(())
}

fn expand_gpu_fn_impl(
    attr: TokenStream2,
    item: TokenStream2,
    same_name_callable: bool,
) -> syn::Result<TokenStream2> {
    let item_fn: ItemFn = syn::parse2(item)?;
    let sig = &item_fn.sig;
    validate_fn_common_signature(sig)?;

    // Determine whether this is a pure scalar helper function (`fn(f32, ...) -> f32`)
    // or a component kernel function (`fn(&mut CompA, &CompB)`).
    if let ReturnType::Type(_, ret_ty) = &sig.output {
        if !same_name_callable || !attr.is_empty() {
            return Err(Error::new_spanned(
                ret_ty,
                "`#[gpu_kernel]` entry points must not have a return type; use `#[gpu]` for composable scalar helper functions",
            ));
        }
        return expand_gpu_helper_fn_impl(&item_fn, ret_ty);
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

    let root_names: Vec<String> = params.iter().map(|p| p.ident.to_string()).collect();
    let mut ir_builder = IrAstBuilder::new(root_names, false);
    let ir_block = ir_builder.lower_block(&item_fn.block)?;

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
                @ir
                #(#attrs)*
                #vis fn #fn_name ( #(#param_tokens),* )
                ir = #ir_block,
                body = { #(#stmts)* }
            }
        })
    } else {
        let custom_ident: Ident = syn::parse2(attr)?;
        Ok(quote! {
            ::sumeru::sumeru_gpu_kernel! {
                @ir
                #(#attrs)*
                #vis kernel #custom_ident fn #fn_name ( #(#param_tokens),* )
                ir = #ir_block,
                body = { #(#stmts)* }
            }
        })
    }
}

fn expand_gpu_helper_fn_impl(item_fn: &ItemFn, ret_ty: &Type) -> syn::Result<TokenStream2> {
    let sig = &item_fn.sig;
    let ret_kind = parse_scalar_kind_tokens(ret_ty)?;

    let mut params = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(Error::new_spanned(
                arg,
                "`#[gpu]` helper functions cannot take `self`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(Error::new_spanned(
                &pat_type.pat,
                "`#[gpu]` helper parameter pattern must be a simple identifier",
            ));
        };
        let pname = pat_ident.ident.to_string();
        if pname.starts_with("__") || pname.starts_with("sumeru_") {
            return Err(Error::new_spanned(
                &pat_ident.ident,
                "`#[gpu]` parameter name cannot start with reserved prefix `__` or `sumeru_`",
            ));
        }
        if matches!(&*pat_type.ty, Type::Reference(_)) {
            return Err(Error::new_spanned(
                &pat_type.ty,
                format!(
                    "`#[gpu]` function with return type cannot take component reference `{pname}`; component kernels must not return a value, and pure helper functions must take scalar values (`f32`, `i32`, `u32`, `usize`, `bool`)"
                ),
            ));
        }
        let kind_tokens = parse_scalar_kind_tokens(&pat_type.ty)?;
        params.push(ParsedHelperParam {
            ident: pat_ident.ident.clone(),
            is_mut: pat_ident.mutability.is_some(),
            ty: (*pat_type.ty).clone(),
            kind_tokens,
        });
    }

    if params.is_empty() || params.len() > 8 {
        return Err(Error::new_spanned(
            &sig.inputs,
            "`#[gpu]` pure helper functions must have between 1 and 8 scalar parameters",
        ));
    }

    let root_names: Vec<String> = params.iter().map(|p| p.ident.to_string()).collect();
    let mut ir_builder = IrAstBuilder::new(root_names, true);
    let ir_block = ir_builder.lower_block_with_tail_return(&item_fn.block)?;

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
        let ty = &p.ty;
        let kind = &p.kind_tokens;
        quote! { #ident : [#mode] #ty => #kind }
    });

    Ok(quote! {
        ::sumeru::sumeru_gpu_helper_fn! {
            #(#attrs)*
            #vis fn #fn_name ( #(#param_tokens),* ) -> #ret_ty => #ret_kind,
            ir = #ir_block,
            body = { #(#stmts)* }
        }
    })
}

fn parse_scalar_kind_tokens(ty: &Type) -> syn::Result<TokenStream2> {
    let s = ty.to_token_stream().to_string();
    match s.trim() {
        "f32" => Ok(quote! { ::sumeru::GpuScalarKind::F32 }),
        "i32" => Ok(quote! { ::sumeru::GpuScalarKind::I32 }),
        "u32" | "usize" => Ok(quote! { ::sumeru::GpuScalarKind::U32 }),
        "bool" => Ok(quote! { ::sumeru::GpuScalarKind::Bool }),
        _ => Err(Error::new_spanned(
            ty,
            format!(
                "type `{s}` is not supported in `#[gpu]` scalar annotations/casts/helpers; use `f32`, `i32`, `u32`, `usize`, or `bool`"
            ),
        )),
    }
}

/// Lexical-scope-aware AST-to-Typed-Shader-IR builder.
struct IrAstBuilder {
    scopes: Vec<Vec<String>>,
    is_helper_fn: bool,
}

impl IrAstBuilder {
    fn new(root_params: Vec<String>, is_helper_fn: bool) -> Self {
        Self {
            scopes: vec![root_params],
            is_helper_fn,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_local(&mut self, name: String) {
        if let Some(top) = self.scopes.last_mut() {
            top.push(name);
        }
    }

    fn is_in_local_scope(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.iter().any(|v| v == name))
    }

    fn lower_block(&mut self, block: &Block) -> syn::Result<TokenStream2> {
        self.push_scope();
        let mut stmt_tokens = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmt_tokens.push(self.lower_stmt(stmt)?);
        }
        self.pop_scope();
        Ok(quote! {
            ::sumeru::GpuBlock::new(::std::vec![ #(#stmt_tokens),* ])
        })
    }

    fn lower_block_with_tail_return(&mut self, block: &Block) -> syn::Result<TokenStream2> {
        self.push_scope();
        let mut stmt_tokens = Vec::with_capacity(block.stmts.len());
        let len = block.stmts.len();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            if idx + 1 == len {
                stmt_tokens.push(self.lower_tail_stmt(stmt)?);
            } else {
                stmt_tokens.push(self.lower_stmt(stmt)?);
            }
        }
        self.pop_scope();
        Ok(quote! {
            ::sumeru::GpuBlock::new(::std::vec![ #(#stmt_tokens),* ])
        })
    }

    fn lower_tail_stmt(&mut self, stmt: &Stmt) -> syn::Result<TokenStream2> {
        match stmt {
            Stmt::Expr(expr, None) => self.lower_tail_expr(expr),
            _ => self.lower_stmt(stmt),
        }
    }

    fn lower_tail_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Return(_) => self.lower_stmt_expr(expr, true),
            Expr::If(expr_if) => {
                let cond = self.lower_expr(&expr_if.cond)?;
                let then_block = self.lower_block_with_tail_return(&expr_if.then_branch)?;
                let else_tokens = if let Some((_, else_expr)) = &expr_if.else_branch {
                    match &**else_expr {
                        Expr::If(_) => {
                            let elif = self.lower_tail_expr(else_expr)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseIf(::std::boxed::Box::new(#elif))) }
                        }
                        Expr::Block(eb) => {
                            let eblock = self.lower_block_with_tail_return(&eb.block)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseBlock(#eblock)) }
                        }
                        other => {
                            return Err(Error::new_spanned(
                                other,
                                "unsupported `else` branch in `#[gpu]` function",
                            ));
                        }
                    }
                } else {
                    quote! { ::std::option::Option::None }
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::If {
                        cond: #cond,
                        then_block: #then_block,
                        else_branch: #else_tokens,
                    }
                })
            }
            Expr::Block(eb) => {
                if eb.label.is_some() {
                    return Err(Error::new_spanned(
                        &eb.label,
                        "labeled blocks are not supported in `#[gpu]` functions",
                    ));
                }
                let inner = self.lower_block_with_tail_return(&eb.block)?;
                Ok(quote! { ::sumeru::GpuStmt::Block(#inner) })
            }
            other => {
                let val = self.lower_expr(other)?;
                Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::Some(#val)) })
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> syn::Result<TokenStream2> {
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
                        let kind_tokens = parse_scalar_kind_tokens(&pt.ty)?;
                        (&pi.ident, pi.mutability.is_some(), Some(kind_tokens))
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
                let rhs = self.lower_expr(&init.expr)?;
                self.declare_local(vname.clone());
                let explicit_ty_tokens = match explicit_ty {
                    Some(k) => quote! { ::std::option::Option::Some(#k) },
                    None => quote! { ::std::option::Option::None },
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::Let {
                        name: #vname,
                        mutable: #is_mut,
                        explicit_ty: #explicit_ty_tokens,
                        init: #rhs,
                    }
                })
            }
            Stmt::Item(Item::Const(item_const)) => {
                let vname = item_const.ident.to_string();
                if vname.starts_with("__") || vname.starts_with("sumeru_") {
                    return Err(Error::new_spanned(
                        &item_const.ident,
                        "const name in `#[gpu]` kernel cannot start with `__` or `sumeru_`",
                    ));
                }
                let kind_tokens = parse_scalar_kind_tokens(&item_const.ty)?;
                let rhs = self.lower_expr(&item_const.expr)?;
                self.declare_local(vname.clone());
                Ok(quote! {
                    ::sumeru::GpuStmt::Let {
                        name: #vname,
                        mutable: false,
                        explicit_ty: ::std::option::Option::Some(#kind_tokens),
                        init: #rhs,
                    }
                })
            }
            Stmt::Item(item) => Err(Error::new_spanned(
                item,
                "nested item declarations (other than scalar `const`) are not supported in `#[gpu]` kernels",
            )),
            Stmt::Macro(mac) => Err(Error::new_spanned(
                mac,
                "macros are not supported inside `#[gpu]` kernel bodies",
            )),
            Stmt::Expr(expr, semi) => self.lower_stmt_expr(expr, semi.is_some()),
        }
    }

    fn lower_stmt_expr(&mut self, expr: &Expr, has_semi: bool) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Assign(assign) => {
                let lhs = self.lower_lvalue(&assign.left)?;
                let rhs = self.lower_expr(&assign.right)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::Assign {
                        op: ::sumeru::GpuAssignOp::Assign,
                        target: #lhs,
                        value: #rhs,
                    }
                })
            }
            Expr::Binary(bin) if is_compound_assign(&bin.op) => {
                let lhs = self.lower_lvalue(&bin.left)?;
                let op = lower_assign_op(&bin.op)?;
                let rhs = self.lower_expr(&bin.right)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::Assign {
                        op: #op,
                        target: #lhs,
                        value: #rhs,
                    }
                })
            }
            Expr::If(expr_if) => {
                let cond = self.lower_expr(&expr_if.cond)?;
                let then_block = self.lower_block(&expr_if.then_branch)?;
                let else_tokens = if let Some((_, else_expr)) = &expr_if.else_branch {
                    match &**else_expr {
                        Expr::If(_) => {
                            let else_if = self.lower_stmt_expr(else_expr, true)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseIf(::std::boxed::Box::new(#else_if))) }
                        }
                        Expr::Block(eb) => {
                            let else_body = self.lower_block(&eb.block)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseBlock(#else_body)) }
                        }
                        other => {
                            return Err(Error::new_spanned(
                                other,
                                "unsupported `else` branch in `#[gpu]` kernel",
                            ));
                        }
                    }
                } else {
                    quote! { ::std::option::Option::None }
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::If {
                        cond: #cond,
                        then_block: #then_block,
                        else_branch: #else_tokens,
                    }
                })
            }
            Expr::While(expr_while) => {
                let cond = self.lower_expr(&expr_while.cond)?;
                let body = self.lower_block(&expr_while.body)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::While {
                        cond: #cond,
                        body: #body,
                    }
                })
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
                let start = self.lower_expr(start_expr)?;
                let end = self.lower_expr(end_expr)?;
                let inclusive = matches!(range.limits, RangeLimits::Closed(_));
                self.push_scope();
                self.declare_local(var_name.clone());
                let body = self.lower_block(&expr_for.body)?;
                self.pop_scope();
                Ok(quote! {
                    ::sumeru::GpuStmt::ForRange {
                        var: #var_name,
                        start: #start,
                        end: #end,
                        inclusive: #inclusive,
                        body: #body,
                    }
                })
            }
            Expr::Block(eb) => {
                if eb.label.is_some() {
                    return Err(Error::new_spanned(
                        &eb.label,
                        "labeled blocks are not supported in `#[gpu]` kernels",
                    ));
                }
                let inner = self.lower_block(&eb.block)?;
                Ok(quote! { ::sumeru::GpuStmt::Block(#inner) })
            }
            Expr::Return(ret) => {
                if let Some(val) = &ret.expr {
                    if !self.is_helper_fn {
                        return Err(Error::new_spanned(
                            val,
                            "`#[gpu]` kernel cannot return a value",
                        ));
                    }
                    let lowered_val = self.lower_expr(val)?;
                    Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::Some(#lowered_val)) })
                } else {
                    if self.is_helper_fn {
                        return Err(Error::new_spanned(
                            ret,
                            "`#[gpu]` helper function with a return type must return a value",
                        ));
                    }
                    Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::None) })
                }
            }
            Expr::Break(brk) => {
                if brk.label.is_some() || brk.expr.is_some() {
                    return Err(Error::new_spanned(
                        brk,
                        "labeled or value `break` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok(quote! { ::sumeru::GpuStmt::Break })
            }
            Expr::Continue(cont) => {
                if cont.label.is_some() {
                    return Err(Error::new_spanned(
                        cont,
                        "labeled `continue` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok(quote! { ::sumeru::GpuStmt::Continue })
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

    fn lower_lvalue(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
                let name = p.path.segments[0].ident.to_string();
                Ok(quote! { ::sumeru::GpuLValue::Var(#name) })
            }
            Expr::Field(f) => {
                let base = self.lower_lvalue(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuLValue::Field {
                        base: ::std::boxed::Box::new(#base),
                        field: #member,
                    }
                })
            }
            Expr::Index(idx) => {
                let base = self.lower_lvalue(&idx.expr)?;
                let index = self.lower_expr(&idx.index)?;
                Ok(quote! {
                    ::sumeru::GpuLValue::Index {
                        base: ::std::boxed::Box::new(#base),
                        index: ::std::boxed::Box::new(#index),
                    }
                })
            }
            Expr::Unary(un) if matches!(un.op, UnOp::Deref(_)) => self.lower_lvalue(&un.expr),
            Expr::Paren(p) => self.lower_lvalue(&p.expr),
            other => Err(Error::new_spanned(
                other,
                "invalid assignment target in `#[gpu]` kernel",
            )),
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Path(p) => {
                if p.qself.is_none() && p.path.segments.len() == 1 {
                    let ident_str = p.path.segments[0].ident.to_string();
                    if self.is_in_local_scope(&ident_str) {
                        return Ok(quote! { ::sumeru::GpuExpr::Var(#ident_str) });
                    }
                }
                // External Rust `const` identifier or multi-segment path (e.g. `DT` or `std::f32::consts::PI`):
                // evaluated at compile time by `rustc` via `GpuConstValue`.
                Ok(quote! { ::sumeru::GpuConstValue::to_gpu_expr(&(#p)) })
            }
            Expr::Field(f) => {
                let base = self.lower_expr(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuExpr::Field {
                        base: ::std::boxed::Box::new(#base),
                        field: #member,
                    }
                })
            }
            Expr::Index(idx) => {
                let base = self.lower_expr(&idx.expr)?;
                let index = self.lower_expr(&idx.index)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Index {
                        base: ::std::boxed::Box::new(#base),
                        index: ::std::boxed::Box::new(#index),
                    }
                })
            }
            Expr::Binary(bin) => {
                if is_compound_assign(&bin.op) {
                    return Err(Error::new_spanned(
                        bin,
                        "compound assignment cannot be used as a sub-expression in `#[gpu]` kernels",
                    ));
                }
                let lhs = self.lower_expr(&bin.left)?;
                let op = lower_binop(&bin.op)?;
                let rhs = self.lower_expr(&bin.right)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Binary {
                        op: #op,
                        lhs: ::std::boxed::Box::new(#lhs),
                        rhs: ::std::boxed::Box::new(#rhs),
                    }
                })
            }
            Expr::Unary(un) => match un.op {
                UnOp::Neg(_) => {
                    let inner = self.lower_expr(&un.expr)?;
                    Ok(quote! {
                        ::sumeru::GpuExpr::Unary {
                            op: ::sumeru::GpuUnaryOp::Neg,
                            expr: ::std::boxed::Box::new(#inner),
                        }
                    })
                }
                UnOp::Not(_) => {
                    let inner = self.lower_expr(&un.expr)?;
                    Ok(quote! {
                        ::sumeru::GpuExpr::Unary {
                            op: ::sumeru::GpuUnaryOp::Not,
                            expr: ::std::boxed::Box::new(#inner),
                        }
                    })
                }
                UnOp::Deref(_) => self.lower_expr(&un.expr),
                _ => Err(Error::new_spanned(
                    un,
                    "unsupported unary operator in `#[gpu]` kernel",
                )),
            },
            Expr::Paren(p) => {
                let inner = self.lower_expr(&p.expr)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Paren(::std::boxed::Box::new(#inner))
                })
            }
            Expr::Cast(cast) => {
                let inner = self.lower_expr(&cast.expr)?;
                let ty = parse_scalar_kind_tokens(&cast.ty)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Cast {
                        expr: ::std::boxed::Box::new(#inner),
                        ty: #ty,
                    }
                })
            }
            Expr::Lit(expr_lit) => self.lower_lit(&expr_lit.lit),
            Expr::MethodCall(mc) => {
                let receiver = self.lower_expr(&mc.receiver)?;
                let method = mc.method.to_string();
                let mut args = Vec::with_capacity(mc.args.len());
                for arg in &mc.args {
                    args.push(self.lower_expr(arg)?);
                }
                let intrinsic = match (method.as_str(), args.len()) {
                    ("abs", 0) => quote! { ::sumeru::GpuMathIntrinsic::Abs },
                    ("sqrt", 0) => quote! { ::sumeru::GpuMathIntrinsic::Sqrt },
                    ("sin", 0) => quote! { ::sumeru::GpuMathIntrinsic::Sin },
                    ("cos", 0) => quote! { ::sumeru::GpuMathIntrinsic::Cos },
                    ("tan", 0) => quote! { ::sumeru::GpuMathIntrinsic::Tan },
                    ("floor", 0) => quote! { ::sumeru::GpuMathIntrinsic::Floor },
                    ("ceil", 0) => quote! { ::sumeru::GpuMathIntrinsic::Ceil },
                    ("round", 0) => quote! { ::sumeru::GpuMathIntrinsic::Round },
                    ("fract", 0) => quote! { ::sumeru::GpuMathIntrinsic::Fract },
                    ("exp", 0) => quote! { ::sumeru::GpuMathIntrinsic::Exp },
                    ("ln", 0) => quote! { ::sumeru::GpuMathIntrinsic::Ln },
                    ("min", 1) => quote! { ::sumeru::GpuMathIntrinsic::Min },
                    ("max", 1) => quote! { ::sumeru::GpuMathIntrinsic::Max },
                    ("atan2", 1) => quote! { ::sumeru::GpuMathIntrinsic::Atan2 },
                    ("powf", 1) => quote! { ::sumeru::GpuMathIntrinsic::Powf },
                    ("clamp", 2) => quote! { ::sumeru::GpuMathIntrinsic::Clamp },
                    ("mul_add" | "fma", 2) => quote! { ::sumeru::GpuMathIntrinsic::Fma },
                    _ => {
                        return Err(Error::new_spanned(
                            &mc.method,
                            format!(
                                "method `{method}` (with {} arg(s)) is not supported in `#[gpu]` kernels; supported methods: abs, sqrt, min, max, clamp, sin, cos, tan, floor, ceil, round, fract, exp, ln, powf, atan2, mul_add",
                                args.len()
                            ),
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuExpr::MathCall {
                        receiver: ::std::boxed::Box::new(#receiver),
                        intrinsic: #intrinsic,
                        args: ::std::vec![ #(#args),* ],
                    }
                })
            }
            Expr::Call(call) => {
                let Expr::Path(func_path) = &*call.func else {
                    return Err(Error::new_spanned(
                        &call.func,
                        "only direct calls to `#[gpu]` pure helper functions are supported in `#[gpu]` kernels",
                    ));
                };
                if func_path.qself.is_some() {
                    return Err(Error::new_spanned(
                        func_path,
                        "qualified `<T as Trait>::fn` calls are not supported in `#[gpu]` kernels",
                    ));
                }
                let mut args = Vec::with_capacity(call.args.len());
                for arg in &call.args {
                    args.push(self.lower_expr(arg)?);
                }
                let path = &func_path.path;
                Ok(quote! {
                    ::sumeru::GpuExpr::HelperCall {
                        helper: #path::gpu_helper_descriptor(),
                        args: ::std::vec![ #(#args),* ],
                    }
                })
            }
            other => Err(Error::new_spanned(
                other,
                "unsupported Rust expression in `#[gpu]` kernel; closures, `match`, and references are not permitted in GPU shaders",
            )),
        }
    }

    fn lower_lit(&self, lit: &Lit) -> syn::Result<TokenStream2> {
        match lit {
            Lit::Float(fl) => {
                let digits = fl.base10_digits().replace('_', "");
                let formatted = if digits.contains('.') || digits.contains('e') || digits.contains('E') {
                    digits
                } else {
                    format!("{digits}.0")
                };
                Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::FloatRaw(#formatted)) })
            }
            Lit::Int(il) => {
                let suffix = il.suffix();
                match suffix {
                    "" => {
                        let val: i64 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::UntypedInt(#val)) })
                    }
                    "i32" => {
                        let val: i32 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::I32(#val)) })
                    }
                    "u32" | "usize" => {
                        let val: u64 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::U32(#val)) })
                    }
                    other => Err(Error::new_spanned(
                        il,
                        format!(
                            "integer literal suffix `{other}` is not supported in `#[gpu]` kernels; use `i32`, `u32`, or `usize`"
                        ),
                    )),
                }
            }
            Lit::Bool(b) => {
                let val = b.value;
                Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::Bool(#val)) })
            }
            other => Err(Error::new_spanned(
                other,
                "string/byte/char literals are not supported in `#[gpu]` kernels",
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

fn lower_assign_op(op: &BinOp) -> syn::Result<TokenStream2> {
    match op {
        BinOp::AddAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::AddAssign }),
        BinOp::SubAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::SubAssign }),
        BinOp::MulAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::MulAssign }),
        BinOp::DivAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::DivAssign }),
        BinOp::RemAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::RemAssign }),
        BinOp::BitXorAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitXorAssign }),
        BinOp::BitAndAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitAndAssign }),
        BinOp::BitOrAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitOrAssign }),
        BinOp::ShlAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::ShlAssign }),
        BinOp::ShrAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::ShrAssign }),
        other => Err(Error::new_spanned(
            other,
            "unsupported compound assignment operator in `#[gpu]` kernel",
        )),
    }
}

fn lower_binop(op: &BinOp) -> syn::Result<TokenStream2> {
    match op {
        BinOp::Add(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Add }),
        BinOp::Sub(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Sub }),
        BinOp::Mul(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Mul }),
        BinOp::Div(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Div }),
        BinOp::Rem(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Rem }),
        BinOp::And(_) => Ok(quote! { ::sumeru::GpuBinaryOp::And }),
        BinOp::Or(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Or }),
        BinOp::BitXor(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitXor }),
        BinOp::BitAnd(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitAnd }),
        BinOp::BitOr(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitOr }),
        BinOp::Shl(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Shl }),
        BinOp::Shr(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Shr }),
        BinOp::Eq(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Eq }),
        BinOp::Lt(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Lt }),
        BinOp::Le(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Le }),
        BinOp::Ne(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Ne }),
        BinOp::Ge(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Ge }),
        BinOp::Gt(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Gt }),
        other => Err(Error::new_spanned(
            other,
            "unsupported binary operator in `#[gpu]` kernel",
        )),
    }
}
