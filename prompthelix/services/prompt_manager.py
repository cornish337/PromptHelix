import uuid
from typing import Dict, List, Optional

class PromptManager:
    """Simple in-memory manager for storing prompts."""

    def __init__(self) -> None:
        self._prompts: Dict[str, str] = {}

    def add_prompt(self, content: str) -> dict:
        prompt_id = str(uuid.uuid4())
        self._prompts[prompt_id] = content
        return {"id": prompt_id, "content": content}

    def get_prompt(self, prompt_id: str) -> Optional[str]:
        return self._prompts.get(prompt_id)

    def list_prompts(self) -> List[dict]:
        return [{"id": pid, "content": content} for pid, content in self._prompts.items()]

