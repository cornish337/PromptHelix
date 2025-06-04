from typing import Dict, List
from uuid import uuid4

class PromptManager:
    """Simple in-memory manager for prompts."""

    def __init__(self) -> None:
        self._prompts: Dict[str, str] = {}

    def add_prompt(self, content: str) -> Dict[str, str]:
        """Add a new prompt and return its data."""
        prompt_id = str(uuid4())
        self._prompts[prompt_id] = content
        return {"id": prompt_id, "content": content}

    def get_prompt(self, prompt_id: str) -> str | None:
        """Retrieve a prompt by ID."""
        return self._prompts.get(prompt_id)

    def list_prompts(self) -> List[Dict[str, str]]:
        """Return all prompts in a serializable form."""
        return [
            {"id": pid, "content": content} for pid, content in self._prompts.items()
        ]
