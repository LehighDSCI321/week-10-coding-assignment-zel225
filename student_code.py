

"""Week 10: Graph Traversal (BFS/DFS) and DAG."""
from typing import Any, Deque, Dict, List, Set, Tuple
from collections import deque
from collections.abc import Hashable

class VersatileDigraph:
    """A lightweight directed graph.

    Internally, the graph is an adjacency dict:
    node -> set of outgoing neighbors.
    Nodes must be hashable (e.g., strings used in tests).
    """

    def __init__(self) -> None:
        """Create an empty directed graph."""
        self._adj: Dict[Hashable, Set[Hashable]] = {}

    # ------------- Mutators -------------------------------------------------
    def add_node(self, node: Hashable) -> None:
        """Add a node if it does not exist."""
        if node not in self._adj:
            self._adj[node] = set()

    def add_edge(self, source: Hashable, target: Hashable) -> None:
        """Add a directed edge source -> target. Nodes are created if missing."""
        self.add_node(source)
        self.add_node(target)
        self._adj[source].add(target)

    # ------------- Basic queries -------------------------------------------
    def nodes(self) -> List[Hashable]:
        """Return all nodes. If nodes are comparable, return sorted for stability."""
        try:
            return sorted(self._adj.keys())
        except TypeError:
            return list(self._adj.keys())

    def edges(self) -> List[Tuple[Hashable, Hashable]]:
        """Return all directed edges as (u, v) pairs."""
        res: List[Tuple[Hashable, Hashable]] = []
        for u, nbrs in self._adj.items():
            for v in nbrs:
                res.append((u, v))
        return res

    def get_neighbors(self, node: Hashable) -> Set[Hashable]:
        """Return the outgoing neighbors of a node (copy)."""
        return set(self._adj.get(node, set()))

    def neighbors(self, node: Hashable) -> Set[Hashable]:
        """Alias of get_neighbors for convenience."""
        return self.get_neighbors(node)

    def __contains__(self, node: Hashable) -> bool:  # pragma: no cover
        """Return True if node exists in the graph."""
        return node in self._adj

    def __len__(self) -> int:  # pragma: no cover
        """Return number of nodes."""
        return len(self._adj)

    # ------------- Internal helpers ----------------------------------------
    def _indegree_map(self) -> Dict[Hashable, int]:
        """Compute indegree for each node."""
        indeg: Dict[Hashable, int] = {n: 0 for n in self._adj}
        for _, nbrs in self._adj.items():
            for v in nbrs:
                indeg[v] = indeg.get(v, 0) + 1
        return indeg


class SortableDigraph(VersatileDigraph):
    """A digraph that supports topological sorting."""

    def top_sort(self) -> List[Hashable]:
        """Return a topological order of the nodes.

        Kahn's algorithm:
        1) Compute indegrees.
        2) Start with all nodes with indegree 0.
        3) Repeatedly pop a ready node, append to output, relax its edges.
        4) If nodes remain with positive indegree, a cycle exists.

        """
        if not self._adj:
            return []

        indeg = self._indegree_map()

        # Initial ready set: indegree == 0
        ready: List[Hashable] = [n for n, d in indeg.items() if d == 0]
        self._safe_sort_in_place(ready)

        order: List[Hashable] = []
        while ready:
            current = ready.pop(0)
            order.append(current)

            nbrs = list(self._adj.get(current, set()))
            self._safe_sort_in_place(nbrs)

            for v in nbrs:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
                    self._safe_sort_in_place(ready)

        if len(order) != len(indeg):
            raise ValueError("Graph contains a cycle; topological sort is undefined.")

        return order

    # ---------------- Private utility --------------------------------------
    @staticmethod
    def _safe_sort_in_place(items: List[Hashable]) -> None:
        """Sort items in place if comparable; otherwise keep order."""
        try:
            items.sort()  # type: ignore[call-arg]
        except TypeError:
            pass

class TraversableDigraph(SortableDigraph):
    """A digraph that supports traversal (BFS/DFS) and simple metadata.

    Additions over SortableDigraph:
      - optional node values
      - optional edge weights
      - ordered successors / predecessors
      - bfs / dfs generators that YIELD reachable nodes (excluding the start)
    """

    def __init__(self) -> None:
        super().__init__()
        self._values: Dict[Hashable, Any] = {}
        # store weights in a flat dict keyed by (u, v)
        self._weights: Dict[Tuple[Hashable, Hashable], float] = {}

    # ---------- node & edge metadata ----------
    def add_node(self, node: Hashable, value: Any | None = None) -> None:
        """Add node and (optionally) record its value."""
        if node not in self._adj:
            self._adj[node] = set()
        if value is not None:
            self._values[node] = value

    def get_nodes(self) -> List[Hashable]:
        """Alias returning all nodes (sorted if possible)."""
        return self.nodes()

    def get_node_value(self, node: Hashable) -> Any | None:
        """Return the stored value for a node (or None if absent)."""
        return self._values.get(node, None)

    def add_edge(
        self, source: Hashable, target: Hashable, edge_weight: float = 1.0
    ) -> None:
        """Add edge and (optionally) record its weight."""
        super().add_edge(source, target)
        self._weights[(source, target)] = edge_weight

    def get_edge_weight(self, source: Hashable, target: Hashable) -> float:
        """Return the stored edge weight. Raises KeyError if edge missing."""
        if (source, target) not in self._weights:
            # fall back to presence in adjacency to give a clearer error
            if target not in self._adj.get(source, set()):
                raise KeyError(f"Edge {source!r}->{target!r} does not exist.")
            # edge exists but no explicit weight recorded => default 1.0
            return 1.0
        return self._weights[(source, target)]

    # ---------- neighborhood queries ----------
    def successors(self, node: Hashable) -> List[Hashable]:
        """Return the direct successors of node as a (deterministically) ordered list."""
        nbrs = list(self._adj.get(node, set()))
        self._safe_sort_in_place(nbrs)
        return nbrs

    def predecessors(self, node: Hashable) -> List[Hashable]:
        """Return the direct predecessors of node as a (deterministically) ordered list."""
        preds: List[Hashable] = []
        for u, nbrs in self._adj.items():
            if node in nbrs:
                preds.append(u)
        self._safe_sort_in_place(preds)
        return preds

    # ---------- traversals ----------
    def bfs(self, start: Hashable):
        """Breadth-first traversal. Yield reachable nodes, EXCLUDING the start.

        Uses a deque for O(1) pops from the left.
        Children are visited in sorted order when items are comparable.
        """
        if start not in self._adj:
            return
        visited: Set[Hashable] = {start}
        q: Deque[Hashable] = deque([start])

        while q:
            u = q.popleft()
            nbrs = self.successors(u)  # already deterministically ordered
            for v in nbrs:
                if v not in visited:
                    visited.add(v)
                    # NOTE: do not yield the start itself
                    yield v
                    q.append(v)

    def dfs(self, start: Hashable):
        """Depth-first traversal. Yield reachable nodes, EXCLUDING the start.

        Implemented as an explicit stack; children are pushed in reverse
        sorted order so that the smallest neighbor is processed first.
        """
        if start not in self._adj:
            return
        visited: Set[Hashable] = {start}
        stack: List[Tuple[Hashable, List[Hashable]]] = [(start, self.successors(start))]

        while stack:
            _, children = stack[-1]
            if not children:
                stack.pop()
                continue
            # pop next child to explore
            v = children.pop(0)
            if v in visited:
                continue
            visited.add(v)
            # first time we see v -> yield it (exclude start)
            yield v
            # push v with its (ordered) children
            stack.append((v, self.successors(v)))


class DAG(TraversableDigraph):
    """A Directed Acyclic Graph that rejects edges which would create a cycle."""

    def _reachable(self, src: Hashable, dst: Hashable) -> bool:
        """Return True if there exists a path src -> ... -> dst."""
        if src == dst:
            return True
        if src not in self._adj or dst not in self._adj:
            return False
        seen: Set[Hashable] = {src}
        q: Deque[Hashable] = deque([src])
        while q:
            u = q.popleft()
            for v in self.successors(u):
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def add_edge(
        self, source: Hashable, target: Hashable, edge_weight: float = 1.0
    ) -> None:
        """Add edge only if it would NOT create a cycle.

        If there is already a path target -> ... -> source, then adding (source, target)
        would create a cycle, so we raise ValueError.
        Self-loops are also rejected.
        """
        if source == target:
            raise ValueError("Self-loop would create a cycle in DAG.")
        if self._reachable(target, source):
            raise ValueError("Adding this edge would create a cycle in DAG.")
        # safe to add
        super().add_edge(source, target, edge_weight=edge_weight)
