"""
Entity Graph.

Maps relationships between entities (people, projects, products, etc).
"""
from __future__ import annotations

from wild_memory.store.base import MemoryStore


class EntityGraph:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        display_name: str,
        attributes: dict | None = None,
    ) -> None:
        await self.store.upsert_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            display_name=display_name,
            attributes=attributes or {},
        )

    async def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_obs: str | None = None,
        properties: dict | None = None,
    ) -> None:
        await self.store.upsert_edge(
            subject_id=subject,
            predicate=predicate,
            object_id=obj,
            source_observation=source_obs,
            properties=properties or {},
        )

    async def traverse(self, start_entity: str, max_depth: int = 2) -> dict:
        """Breadth-first traversal returning a subgraph dict."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_entity, 0)]
        subgraph: dict = {"nodes": {}, "edges": []}

        while queue:
            eid, depth = queue.pop(0)
            if eid in visited or depth > max_depth:
                continue
            visited.add(eid)

            node = await self.store.get_entity(eid)
            if node:
                subgraph["nodes"][eid] = node

            edges = await self.store.list_edges_for_entity(eid)
            for e in edges:
                subgraph["edges"].append(e)
                next_id = (
                    e["object_id"] if e["subject_id"] == eid else e["subject_id"]
                )
                queue.append((next_id, depth + 1))

        return subgraph

    async def update_attribute(
        self, entity_id: str, attribute: str, value
    ) -> None:
        node = await self.store.get_entity(entity_id)
        if node:
            attrs = dict(node.get("attributes", {}))
            attrs[attribute] = value
            await self.store.update_entity_attributes(entity_id, attrs)
