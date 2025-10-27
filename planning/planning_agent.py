# planning/planning_agent.py
from collections import deque
import logging

class PlanningAgent:
    def __init__(self):
        pass

    def create_plan(self, dependency_graph: dict, sources: list) -> list:
        logging.info("PlanningAgent: generating migration order...")
        if not dependency_graph:
            return sources

        in_degree = {u: 0 for u in dependency_graph}
        for u, deps in dependency_graph.items():
            for v in deps:
                if v in in_degree:
                    in_degree[v] += 1

        q = deque([u for u in in_degree if in_degree[u] == 0])
        order = []
        rev_adj = {u: [] for u in dependency_graph}
        for u, deps in dependency_graph.items():
            for v in deps:
                if v in rev_adj:
                    rev_adj[v].append(u)

        while q:
            u = q.popleft()
            if u in sources:
                order.append(u)
            for v in rev_adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

        missing = [s for s in sources if s not in order]
        if missing:
            logging.warning(f"{len(missing)} sources added manually (cycle or missing dep).")
            order.extend(missing)
        return order
