//! A fast, arena-backed Monte Carlo tree core for AlphaZero chess.
//!
//! Python (selfplay.py / mcts.py) owns the chess rules (via the Rust `fastchess`
//! backend), board->tensor encoding, the GPU inference server, Dirichlet noise
//! and move selection.  Exactly the hot *numeric* loops are ported here:
//!
//!   - `select_batch`  — PUCT selection with virtual loss + in-batch dedup
//!                       (mirrors mcts.py `_collect_batch` / `_select_child`)
//!   - `backprop`       — value back-propagation that also removes virtual loss
//!
//! Nodes are stored in a dense arena (SoA) with no per-node Python objects and
//! no hash maps; children are append-only vectors keyed by policy index.
//!
//! Exposed as a `cdylib` and driven from Python via `ctypes`.

use std::cell::{Cell, RefCell};
use std::sync::Mutex;

const NONE: i32 = -1;

struct Node {
    parent: i32,
    n: u32,
    w: f64,
    q: f64,
    policy: f32,      // current prior (after Dirichlet, if any)
    policy_orig: f32, // untouched prior (for noise / recycling)
    is_expanded: bool,
    virtual_loss: u16,
    // Policy index (0..NUM_ACTIONS) of the move that produced this node, or
    // NONE for the root.  Lets the Python driver recover the move for a node
    // without mirroring every child as a Python object.
    action: i32,
}

struct Arena {
    nodes: Vec<Node>,
    // children[node] = list of (child_node_id). Node index == its own id.
    children: Vec<Vec<i32>>,
    // Per-search scratch buffers to avoid allocation churn.
    scratch_selected: Vec<i32>,
    scratch_path: Vec<i32>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            nodes: Vec::new(),
            children: Vec::new(),
            scratch_selected: Vec::with_capacity(4096),
            scratch_path: Vec::with_capacity(128),
        }
    }
    fn clear(&mut self) {
        self.nodes.clear();
        self.children.clear();
        self.scratch_selected.clear();
        self.scratch_path.clear();
    }
}

// Arenas are addressed by a small integer handle so several engines can coexist
// in one process/thread (e.g. gating eval runs engine A vs engine B).  The
// handle list only changes on engine create/destroy; the hot path reads the
// calling thread's current arena pointer directly (no lock).
static ARENAS: Mutex<Vec<&'static Mutex<Arena>>> = Mutex::new(Vec::new());

thread_local! {
    static CUR: Cell<*const Mutex<Arena>> = const { Cell::new(std::ptr::null()) };
    static FREE: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

fn arena_at(h: i32) -> Option<&'static Mutex<Arena>> {
    let v = ARENAS.lock().unwrap();
    if h >= 0 && (h as usize) < v.len() {
        Some(v[h as usize])
    } else {
        None
    }
}

fn ensure_default() -> &'static Mutex<Arena> {
    let mut v = ARENAS.lock().unwrap();
    if v.is_empty() {
        v.push(Box::leak(Box::new(Mutex::new(Arena::new()))));
    }
    v[0]
}

/// The calling thread's current arena (falls back to a default slot for the
/// low-level test driver, which never calls `mcts_rs_new_arena`).
fn arena() -> &'static Mutex<Arena> {
    let p = CUR.with(|c| c.get());
    if p.is_null() {
        let a = ensure_default();
        CUR.with(|c| c.set(a as *const _));
        return a;
    }
    unsafe { &*p }
}

fn create_node(a: &mut Arena, parent: i32, policy: f32) -> i32 {
    let id = a.nodes.len() as i32;
    a.nodes.push(Node {
        parent,
        n: 0,
        w: 0.0,
        q: 0.0,
        policy,
        policy_orig: policy,
        is_expanded: false,
        virtual_loss: 0,
        action: NONE,
    });
    a.children.push(Vec::new());
    if parent >= 0 {
        a.children[parent as usize].push(id);
    }
    id
}

/// Select the child of `node` with the highest PUCT score, mirroring
/// mcts.py `_select_child`.  Returns child id, or NONE if no children.
fn select_child(a: &Arena, node: usize, c_puct: f32) -> i32 {
    let nd = &a.nodes[node];
    let sqrt_parent_n = (nd.n as f32 + 1.0).sqrt();
    let mut best_score = f32::NEG_INFINITY;
    let mut best_child: i32 = NONE;
    for &child in &a.children[node] {
        let c = &a.nodes[child as usize];
        let vl = c.virtual_loss as f32;
        let eff_n = c.n as f32 + vl;
        let eff_w = c.w as f32 + vl;
        let eff_q = if eff_n > 0.0 { eff_w / eff_n } else { 0.0 };
        let ucb = -eff_q + c_puct * c.policy * sqrt_parent_n / (1.0 + eff_n);
        if ucb > best_score {
            best_score = ucb;
            best_child = child;
        }
    }
    best_child
}

// ── FFI (cdylib) API ──────────────────────────────────────────────────────

/// Create an empty arena. Safe to call more than once; no-op.
#[no_mangle]
pub extern "C" fn mcts_rs_init() {
    let _ = arena();
}

/// Allocate (or recycle) an arena handle for a new engine instance.  Returns a
/// non-negative handle to pass to `mcts_rs_use_arena` / `mcts_rs_free_arena`.
#[no_mangle]
pub extern "C" fn mcts_rs_new_arena() -> i32 {
    if let Some(h) = FREE.with(|f| f.borrow_mut().pop()) {
        if let Some(a) = arena_at(h) {
            a.lock().unwrap().clear();
        }
        if let Some(a) = arena_at(h) {
            CUR.with(|c| c.set(a as *const _));
        }
        return h;
    }
    let leaked: &'static Mutex<Arena> = Box::leak(Box::new(Mutex::new(Arena::new())));
    let mut v = ARENAS.lock().unwrap();
    let h = v.len() as i32;
    v.push(leaked);
    CUR.with(|c| c.set(leaked as *const _));
    h
}

/// Make `handle` the calling thread's current arena for subsequent calls.
#[no_mangle]
pub extern "C" fn mcts_rs_use_arena(handle: i32) {
    if let Some(a) = arena_at(handle) {
        CUR.with(|c| c.set(a as *const _));
    }
}

/// Return an arena to the free list (cleared on next reuse).
#[no_mangle]
pub extern "C" fn mcts_rs_free_arena(handle: i32) {
    if let Some(a) = arena_at(handle) {
        a.lock().unwrap().clear();
    }
    FREE.with(|f| {
        let mut f = f.borrow_mut();
        if !f.contains(&handle) {
            f.push(handle);
        }
    });
}

/// Reset the arena (drop all nodes). Useful between games if a worker reuses
/// the module.
#[no_mangle]
pub extern "C" fn mcts_rs_reset() {
    let mut a = arena().lock().unwrap();
    a.clear();
}

/// Create the root node for a search. Returns its id.
#[no_mangle]
pub extern "C" fn mcts_rs_root() -> i32 {
    let mut a = arena().lock().unwrap();
    create_node(&mut a, NONE, 0.0)
}

/// Create a child of `parent` with prior probability `policy`. Returns new id.
#[no_mangle]
pub extern "C" fn mcts_rs_add_child(parent: i32, policy: f32) -> i32 {
    let mut a = arena().lock().unwrap();
    if parent < 0 || parent as usize >= a.nodes.len() {
        return NONE;
    }
    create_node(&mut a, parent, policy)
}

/// Mark a node expanded (has children + legal priors assigned).
#[no_mangle]
pub extern "C" fn mcts_rs_mark_expanded(node: i32, expanded: bool) {
    let mut a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].is_expanded = expanded;
    }
}

/// Overwrite a node's current policy (used to apply Dirichlet noise).
#[no_mangle]
pub extern "C" fn mcts_rs_set_policy(node: i32, policy: f32) {
    let mut a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].policy = policy;
    }
}

/// Read a node's current policy (for tests / debugging).
#[no_mangle]
pub extern "C" fn mcts_rs_get_policy(node: i32) -> f32 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].policy
    } else {
        f32::NAN
    }
}

/// Read a node's untouched prior (for Dirichlet noise / recycling).
#[no_mangle]
pub extern "C" fn mcts_rs_get_policy_orig(node: i32) -> f32 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].policy_orig
    } else {
        f32::NAN
    }
}

/// Read back and pop a node's children as ids into `out`. Returns count.
#[no_mangle]
pub extern "C" fn mcts_rs_children(node: i32, out: *mut i32, max_out: usize) -> i32 {
    let a = arena().lock().unwrap();
    if node < 0 || node as usize >= a.nodes.len() {
        return 0;
    }
    let kids = &a.children[node as usize];
    let n = kids.len().min(max_out);
    for (i, &c) in kids.iter().take(n).enumerate() {
        unsafe { *out.add(i) = c };
    }
    n as i32
}

/// Select a batch of leaves from `root` using PUCT + virtual loss, mirroring
/// mcts.py `_collect_batch`. Applies virtual loss along interior paths; the
/// chosen leaves are written to `out_leaves`. Returns count (may be < batch).
#[no_mangle]
pub extern "C" fn mcts_rs_select_batch(
    root: i32,
    c_puct: f32,
    batch_size: i32,
    out_leaves: *mut i32,
    max_out: usize,
    out_depths: *mut i32,
) -> i32 {
    let mut a = arena().lock().unwrap();
    select_batch_impl(&mut a, root, c_puct, batch_size, out_leaves, max_out,
                      out_depths, &[])
}

/// Like `mcts_rs_select_batch`, but never selects a node whose id appears in
/// `exclude` (used by the async inference pipeline to avoid dispatching the
/// same leaf twice while an earlier batch is still in flight).
#[no_mangle]
pub extern "C" fn mcts_rs_select_batch_ex(
    root: i32,
    c_puct: f32,
    batch_size: i32,
    out_leaves: *mut i32,
    max_out: usize,
    out_depths: *mut i32,
    exclude: *const i32,
    exclude_count: usize,
) -> i32 {
    let excl = if exclude.is_null() || exclude_count == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(exclude, exclude_count).to_vec() }
    };
    let mut a = arena().lock().unwrap();
    select_batch_impl(&mut a, root, c_puct, batch_size, out_leaves, max_out,
                      out_depths, &excl)
}

fn select_batch_impl(
    a: &mut Arena,
    root: i32,
    c_puct: f32,
    batch_size: i32,
    out_leaves: *mut i32,
    max_out: usize,
    out_depths: *mut i32,
    exclude: &[i32],
) -> i32 {
    if root < 0 || root as usize >= a.nodes.len() {
        return 0;
    }
    // Minimize lock hold / mem: reuse scratch buffers.
    a.scratch_selected.clear();
    a.scratch_path.clear();

    let target = batch_size.max(0) as usize;
    let video_cap = batch_size.max(0) as usize * 8 + 64; // guard against pathological loops
    let mut attempts = 0usize;
    let mut count = 0usize;

    while count < target && attempts < video_cap {
        attempts += 1;
        let mut node = root;
        a.scratch_path.clear();
        let mut prev = NONE;
        // selection with virtual loss
        loop {
            let nd = &a.nodes[node as usize];
            if !(nd.is_expanded && !a.children[node as usize].is_empty()) {
                break; // reached a leaf (unexpanded, or terminal-expanded w/ no kids)
            }
            if node != prev {
                // apply virtual loss to interior node
                a.nodes[node as usize].virtual_loss += 1;
                a.scratch_path.push(node);
                prev = node;
            }
            let next = select_child(&a, node as usize, c_puct);
            if next == NONE {
                break;
            }
            node = next;
        }

        // dedup: if this leaf was already selected in this batch (or is
        // excluded), roll back the virtual losses we applied to interior
        // nodes and retry.
        if a.scratch_selected.contains(&node) || exclude.contains(&node) {
            let path = std::mem::take(&mut a.scratch_path);
            for n in &path {
                a.nodes[*n as usize].virtual_loss =
                    a.nodes[*n as usize].virtual_loss.saturating_sub(1);
            }
            a.scratch_path = path;
            continue;
        }

        a.scratch_selected.push(node);
        // apply virtual loss to the leaf (mirror of Python: node.virtual_loss += 1)
        a.nodes[node as usize].virtual_loss += 1;
        let depth = a.scratch_path.len() as i32;
        if count < max_out {
            unsafe { *out_leaves.add(count) = node };
        }
        if !out_depths.is_null() && count < max_out {
            unsafe { *out_depths.add(count) = depth };
        }
        count += 1;
    }
    count as i32
}

/// Backpropagate `value` from `node` up the tree, decrementing virtual loss on
/// each visited node and flipping the sign each ply (mirror of
/// `_backpropagate_with_virtual_loss`).
#[no_mangle]
pub extern "C" fn mcts_rs_backprop(node: i32, value: f64) {
    let mut a = arena().lock().unwrap();
    if node < 0 || (node as usize) >= a.nodes.len() {
        return;
    }
    let mut current = node;
    let mut v = value;
    while current != NONE && (current as usize) < a.nodes.len() {
        let cur = &mut a.nodes[current as usize];
        cur.virtual_loss = cur.virtual_loss.saturating_sub(1);
        cur.n += 1;
        cur.w += v;
        cur.q = cur.w / cur.n as f64;
        v = -v;
        let up = cur.parent;
        current = up;
    }
}

/// Promote `node` to be root of a fresh tree (clear parent so backprop stops).
#[no_mangle]
pub extern "C" fn mcts_rs_recycle(node: i32) {
    let mut a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].parent = NONE;
    }
}

/// Return visit count of a node.
#[no_mangle]
pub extern "C" fn mcts_rs_get_n(node: i32) -> i64 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].n as i64
    } else {
        0
    }
}

/// Return total value (W) of a node.
#[no_mangle]
pub extern "C" fn mcts_rs_get_w(node: i32) -> f64 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].w as f64
    } else {
        f64::NAN
    }
}

/// Return current node count in the arena.
#[no_mangle]
pub extern "C" fn mcts_rs_count() -> i64 {
    let a = arena().lock().unwrap();
    a.nodes.len() as i64
}

/// Create all children of `parent` in a single call, then mark `parent`
/// expanded.  `idx` holds the policy index of each legal move and `prior` the
/// (already renormalised) prior; both have `count` entries.  The new child ids
/// are written to `out_ids` in the same order.  Returns `count`.
///
/// This replaces the per-child `mcts_rs_add_child` FFI round-trip (and its
/// per-call lock) that dominated RustMCTS expansion.
#[no_mangle]
pub extern "C" fn mcts_rs_expand_batch(
    parent: i32,
    idx: *const i32,
    prior: *const f32,
    count: usize,
    out_ids: *mut i32,
) -> i32 {
    let mut a = arena().lock().unwrap();
    if parent < 0 || parent as usize >= a.nodes.len() || count == 0 {
        return 0;
    }
    let p = parent as usize;
    for k in 0..count {
        let act = unsafe { *idx.add(k) };
        let pr = unsafe { *prior.add(k) };
        let id = a.nodes.len() as i32;
        a.nodes.push(Node {
            parent,
            n: 0,
            w: 0.0,
            q: 0.0,
            policy: pr,
            policy_orig: pr,
            is_expanded: false,
            virtual_loss: 0,
            action: act,
        });
        a.children.push(Vec::new());
        a.children[p].push(id);
        if !out_ids.is_null() {
            unsafe { *out_ids.add(k) = id };
        }
    }
    a.nodes[p].is_expanded = true;
    count as i32
}

/// Whether a node has been expanded.
#[no_mangle]
pub extern "C" fn mcts_rs_is_expanded(node: i32) -> bool {
    let a = arena().lock().unwrap();
    node >= 0 && (node as usize) < a.nodes.len() && a.nodes[node as usize].is_expanded
}

/// Parent id of `node` (NONE for a root).
#[no_mangle]
pub extern "C" fn mcts_rs_get_parent(node: i32) -> i32 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].parent
    } else {
        NONE
    }
}

/// Policy index of the move that produced `node` (NONE for the root).
#[no_mangle]
pub extern "C" fn mcts_rs_get_action(node: i32) -> i32 {
    let a = arena().lock().unwrap();
    if node >= 0 && (node as usize) < a.nodes.len() {
        a.nodes[node as usize].action
    } else {
        NONE
    }
}

/// Read a node's children (ids + action indices) into `out_ids`/`out_actions`.
/// Returns the number written (capped at `max_out`).
#[no_mangle]
pub extern "C" fn mcts_rs_children2(
    node: i32,
    out_ids: *mut i32,
    out_actions: *mut i32,
    max_out: usize,
) -> i32 {
    let a = arena().lock().unwrap();
    if node < 0 || node as usize >= a.nodes.len() {
        return 0;
    }
    let kids = &a.children[node as usize];
    let n = kids.len().min(max_out);
    for (i, &c) in kids.iter().take(n).enumerate() {
        unsafe {
            if !out_ids.is_null() {
                *out_ids.add(i) = c;
            }
            if !out_actions.is_null() {
                *out_actions.add(i) = a.nodes[c as usize].action;
            }
        }
    }
    n as i32
}