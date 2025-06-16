"""Simple in-memory prompt manager used in unit tests."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

class PromptManager:
    def __init__(self) -> None:
        self._prompts: Dict[str, str] = {}

    def add_prompt(self, content: str) -> Dict[str, str]:
        prompt_id = str(uuid.uuid4())
        self._prompts[prompt_id] = content
        return {"id": prompt_id, "content": content}

    def get_prompt(self, prompt_id: str) -> Optional[str]:
        return self._prompts.get(prompt_id)

    def list_prompts(self) -> List[Dict[str, str]]:
        return [{"id": pid, "content": text} for pid, text in self._prompts.items()]
