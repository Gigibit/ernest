# analysis/semantic_graph.py
import networkx as nx
from typing import List, Tuple

class SemanticGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_nodes(self, nodes: List[str]):
        for n in nodes:
            self.graph.add_node(n)

    def add_edges(self, edges: List[Tuple[str, str, str]]):
        # edges as (from, to, label)
        for frm, to, label in edges:
            self.graph.add_edge(frm, to, label=label)

    def topological_sort(self) -> List[str]:
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # cycle detected
            return list(self.graph.nodes())

    def to_graphml(self, path):
        nx.write_graphml(self.graph, path)
