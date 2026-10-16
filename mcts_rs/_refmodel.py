"""Pure-Python reference model of the Rust MCTS core.

This reproduces mcts.py `_select_child`, `_collect_batch` and
`_backpropagate_with_virtual_loss` in isolation (no chess, no network) so we can
verify the Rust `mcts_rs` arena produces identical results. Used by
`tests/test_mcts_rs.py`.
"""


class RefNode:
    __slots__ = ('parent', 'n', 'w', 'children', 'policy', 'is_expanded',
                 'virtual_loss')

    def __init__(self, parent, policy):
        self.parent = parent
        self.n = 0
        self.w = 0.0
        self.children = []  # list of (policy, node) — matched to rust by index
        self.policy = policy
        self.is_expanded = False
        self.virtual_loss = 0


C_PUCT = 1.5


def select_child(node):
    sqrt_parent_n = (node.n + 1) ** 0.5
    best_score = float('-inf')
    best_child = None
    for child in node.children:
        eff_n = child.n + child.virtual_loss
        eff_w = child.w + child.virtual_loss
        eff_q = eff_w / eff_n if eff_n > 0 else 0.0
        ucb = -eff_q + C_PUCT * child.policy * sqrt_parent_n / (1 + eff_n)
        if ucb > best_score:
            best_score = ucb
            best_child = child
    return best_child


def collect_batch(root, batch_size):
    leaves = []
    selected = set()
    for _ in range(batch_size):
        node = root
        path = []
        while node.is_expanded and node.children:
            node.virtual_loss += 1
            path.append(node)
            node = select_child(node)
        if id(node) in selected:
            for p in path:
                p.virtual_loss -= 1
            continue
        node.virtual_loss += 1
        selected.add(id(node))
        leaves.append(node)
    return leaves


def backprop(node, value):
    cur = node
    v = value
    while cur is not None:
        cur.virtual_loss -= 1
        cur.n += 1
        cur.w += v
        cur = cur.parent
        v = -v