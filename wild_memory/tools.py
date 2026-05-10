"""
Wild Memory — Tool Definitions

Memory tools that the agent can use during conversation. Tool schemas are
provider-agnostic JSON Schema; the active LLMProvider is responsible for
translating them into its native tool format if needed.
"""

RECALL_MEMORY_TOOL = {
    "name": "recall_memory",
    "description": (
        "Search long-term memory for relevant information. Use when you need "
        "context that is not present in the current conversation: prior "
        "history with this user, past decisions, preferences, or details "
        "about a known entity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you want to remember.",
            },
            "search_type": {
                "type": "string",
                "enum": ["semantic", "entity", "temporal"],
                "description": (
                    "Type of search: semantic (by meaning), entity "
                    "(specific person/thing), temporal (recent events)."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID for entity search (e.g., 'person_alex').",
            },
        },
        "required": ["query", "search_type"],
    },
}

SAVE_OBSERVATION_TOOL = {
    "name": "save_observation",
    "description": (
        "Save an important fact to long-term memory immediately. Use when "
        "the user makes a decision, reveals critical information, changes "
        "their mind, expresses a strong preference, or corrects something. "
        "Do NOT use for trivial or already-known information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to save (self-descriptive, standalone).",
            },
            "obs_type": {
                "type": "string",
                "enum": [
                    "decision",
                    "preference",
                    "fact",
                    "goal",
                    "correction",
                    "feedback",
                ],
                "description": "Type of observation.",
            },
            "importance": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "How important (1=trivial, 10=critical).",
            },
            "event_time": {
                "type": "string",
                "description": "When this happened (ISO 8601), if different from now.",
            },
        },
        "required": ["content", "obs_type", "importance"],
    },
}

UPDATE_ENTITY_TOOL = {
    "name": "update_entity",
    "description": (
        "Update information about a known entity (person, project, product, "
        "or other). Use when the user corrects or updates something: "
        "'My name is actually Alex', 'I changed roles', etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "ID of the entity to update.",
            },
            "attribute": {
                "type": "string",
                "description": "Which attribute to update.",
            },
            "new_value": {
                "type": "string",
                "description": "New value for the attribute.",
            },
        },
        "required": ["entity_id", "attribute", "new_value"],
    },
}

MEMORY_TOOLS = [
    RECALL_MEMORY_TOOL,
    SAVE_OBSERVATION_TOOL,
    UPDATE_ENTITY_TOOL,
]
