import pytest
import uuid  # For checking ID format (optional)
from prompthelix.services.prompt_manager import PromptManager

pytest.skip("PromptManager is deprecated; skipping legacy tests", allow_module_level=True)

@pytest.fixture
def manager():
    """Provides a fresh PromptManager instance for each test."""
    return PromptManager()

def test_initialization(manager: PromptManager):
    """Test that a PromptManager instance is created successfully and has an empty internal _prompts dictionary."""
    assert manager._prompts == {}

def test_add_prompt(manager: PromptManager):
    """
    Test that add_prompt returns a dictionary with 'id' and 'content' keys,
    content matches, ID is a string, and prompt is stored.
    """
    content = "This is a test prompt content."
    prompt_data = manager.add_prompt(content)

    assert isinstance(prompt_data, dict)
    assert "id" in prompt_data
    assert "content" in prompt_data
    assert prompt_data["content"] == content
    assert isinstance(prompt_data["id"], str)

    # Optional: Check if ID looks like a UUID
    try:
        uuid.UUID(prompt_data["id"], version=4)
        is_uuid = True
    except ValueError:
        is_uuid = False
    assert is_uuid, "ID should be a valid UUID string"

    # Test that the prompt is actually stored
    retrieved_content = manager.get_prompt(prompt_data["id"])
    assert retrieved_content == content

def test_get_prompt(manager: PromptManager):
    """
    Test retrieving an existing prompt returns the correct content,
    and retrieving a non-existent prompt returns None.
    """
    content = "Content for get_prompt test."
    prompt_data = manager.add_prompt(content)
    prompt_id = prompt_data["id"]

    # Test retrieving an existing prompt
    assert manager.get_prompt(prompt_id) == content

    # Test retrieving a non-existent prompt
    assert manager.get_prompt("non_existent_id") is None

def test_list_prompts_empty(manager: PromptManager):
    """Test that list_prompts returns an empty list when no prompts have been added."""
    assert manager.list_prompts() == []

def test_list_prompts_with_one_prompt(manager: PromptManager):
    """Test that list_prompts returns a list containing the correct prompt data after one prompt is added."""
    content = "A single prompt in the list."
    prompt_data = manager.add_prompt(content)

    expected_list = [{"id": prompt_data["id"], "content": content}]
    assert manager.list_prompts() == expected_list

def test_list_prompts_with_multiple_prompts(manager: PromptManager):
    """Test that list_prompts returns correct data after multiple prompts are added."""
    content1 = "First prompt."
    prompt_data1 = manager.add_prompt(content1)

    content2 = "Second prompt."
    prompt_data2 = manager.add_prompt(content2)

    listed_prompts = manager.list_prompts()
    assert len(listed_prompts) == 2

    # Convert list of dicts to a dict of dicts for easier lookup, as order is not guaranteed
    listed_prompts_dict = {p["id"]: p for p in listed_prompts}

    assert prompt_data1["id"] in listed_prompts_dict
    assert listed_prompts_dict[prompt_data1["id"]]["content"] == content1

    assert prompt_data2["id"] in listed_prompts_dict
    assert listed_prompts_dict[prompt_data2["id"]]["content"] == content2

def test_id_uniqueness(manager: PromptManager):
    """Add two different prompts and check that their returned IDs are different."""
    prompt_data1 = manager.add_prompt("Content A")
    prompt_data2 = manager.add_prompt("Content B")

    assert prompt_data1["id"] != prompt_data2["id"]
