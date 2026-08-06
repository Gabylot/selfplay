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

use std::sync::{Mutex, OnceLock};

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
}

struct Arena {
    nodes: Vec<Node>,
    // children[node] = list of (child_node_id). Node index == its own id.
    children: Vec<Vec<i32>>,
    // This arena is single-threaded per process (one worker per process), so we
    // stash a per-search scratch buffer to avoid allocation churn.
    scratch_selected: Vec<i32>,
    scratch_path: Vec<i32>,
}

fn arena() -> &'static Mutex<Arena> {
    static ARENA: OnceLock<Mutex<Arena>> = OnceLock::new();
    ARENA.get_or_init(|| {
        Mutex::new(Arena {
            nodes: Vec::new(),
            children: Vec::new(),
            scratch_selected: Vec::with_capacity(4096),
            scratch_path: Vec::with_capacity(128),
        })
    })
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

/// Reset the arena (drop all nodes). Useful between games if a worker reuses
/// the module.
#[no_mangle]
pub extern "C" fn mcts_rs_reset() {
    let mut a = arena().lock().unwrap();
    a.nodes.clear();
    a.children.clear();
    a.scratch_selected.clear();
    a.scratch_path.clear();
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

        // dedup: if this leaf was already selected in this batch, roll back
        // the virtual losses we applied to interior nodes and retry.
        if a.scratch_selected.contains(&node) {
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