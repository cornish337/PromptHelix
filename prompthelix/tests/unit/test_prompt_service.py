import pytest
from sqlalchemy.orm import Session as SQLAlchemySession, selectinload # For verifying loaded relationships
from typing import List

from prompthelix.services.prompt_service import PromptService
from prompthelix.schemas import PromptCreate, PromptUpdate, PromptVersionCreate, PromptVersionUpdate
from prompthelix.models.prompt_models import Prompt, PromptVersion

@pytest.fixture(scope="module") # Service instance can be shared across tests in this module
def prompt_service_instance() -> PromptService:
    return PromptService()

def test_create_prompt(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt_data = PromptCreate(name="Test Prompt", description="A test prompt description.")
    db_prompt = prompt_service_instance.create_prompt(db_session, prompt_create=prompt_data)

    assert db_prompt is not None
    assert db_prompt.name == "Test Prompt"
    assert db_prompt.description == "A test prompt description."
    assert db_prompt.id is not None
    assert db_prompt.created_at is not None
    assert len(db_prompt.versions) == 0 # Initially no versions

def test_get_prompt(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    created_prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Get Me"))
    # Add a version to test relationship loading
    prompt_service_instance.create_prompt_version(db_session, created_prompt.id, PromptVersionCreate(content="v1 content"))

    retrieved_prompt = prompt_service_instance.get_prompt(db_session, prompt_id=created_prompt.id)
    assert retrieved_prompt is not None
    assert retrieved_prompt.id == created_prompt.id
    assert retrieved_prompt.name == "Get Me"
    assert len(retrieved_prompt.versions) == 1 # Check if versions are loaded
    assert retrieved_prompt.versions[0].content == "v1 content"

    non_existent_prompt = prompt_service_instance.get_prompt(db_session, prompt_id=9999)
    assert non_existent_prompt is None

def test_get_prompts(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt1 = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Prompt A"))
    prompt_service_instance.create_prompt_version(db_session, prompt1.id, PromptVersionCreate(content="A v1"))
    prompt2 = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Prompt B"))
    prompt_service_instance.create_prompt_version(db_session, prompt2.id, PromptVersionCreate(content="B v1"))
    prompt_service_instance.create_prompt_version(db_session, prompt2.id, PromptVersionCreate(content="B v2"))


    all_prompts = prompt_service_instance.get_prompts(db_session, skip=0, limit=10)
    assert len(all_prompts) == 2
    # Verify versions are loaded for each prompt
    for p in all_prompts:
        if p.id == prompt1.id:
            assert len(p.versions) == 1
        elif p.id == prompt2.id:
            assert len(p.versions) == 2

    # Test pagination
    prompts_limit_1 = prompt_service_instance.get_prompts(db_session, skip=0, limit=1)
    assert len(prompts_limit_1) == 1

    prompts_skip_1 = prompt_service_instance.get_prompts(db_session, skip=1, limit=1)
    assert len(prompts_skip_1) == 1
    assert prompts_skip_1[0].id != prompts_limit_1[0].id # Ensure they are different prompts

def test_update_prompt(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    db_prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Old Name", description="Old Desc"))

    update_data = PromptUpdate(name="New Name", description="New Desc")
    updated_prompt = prompt_service_instance.update_prompt(db_session, prompt_id=db_prompt.id, prompt_update=update_data)
    assert updated_prompt is not None
    assert updated_prompt.name == "New Name"
    assert updated_prompt.description == "New Desc"

    update_partial_name = PromptUpdate(name="Partial Update Name")
    updated_prompt_partial = prompt_service_instance.update_prompt(db_session, prompt_id=db_prompt.id, prompt_update=update_partial_name)
    assert updated_prompt_partial.name == "Partial Update Name"
    assert updated_prompt_partial.description == "New Desc" # Description should persist

    update_non_existent = prompt_service_instance.update_prompt(db_session, prompt_id=9999, prompt_update=PromptUpdate(name="No Such"))
    assert update_non_existent is None

def test_delete_prompt(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt_to_delete = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Delete Me"))
    # Add a version to check cascade delete
    prompt_service_instance.create_prompt_version(db_session, prompt_to_delete.id, PromptVersionCreate(content="v1 content"))

    deleted_prompt = prompt_service_instance.delete_prompt(db_session, prompt_id=prompt_to_delete.id)
    assert deleted_prompt is not None
    assert deleted_prompt.id == prompt_to_delete.id
    assert prompt_service_instance.get_prompt(db_session, prompt_id=prompt_to_delete.id) is None
    # Check if associated versions are also deleted (due to cascade="all, delete-orphan")
    assert db_session.query(PromptVersion).filter(PromptVersion.prompt_id == prompt_to_delete.id).count() == 0

    delete_non_existent = prompt_service_instance.delete_prompt(db_session, prompt_id=9999)
    assert delete_non_existent is None

def test_create_prompt_version(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="Version Test Prompt"))

    version_data1 = PromptVersionCreate(content="Version 1 content", parameters_used={"temp": 0.7}, fitness_score=0.8)
    db_version1 = prompt_service_instance.create_prompt_version(db_session, prompt_id=prompt.id, version_create=version_data1)
    assert db_version1 is not None
    assert db_version1.prompt_id == prompt.id
    assert db_version1.content == "Version 1 content"
    assert db_version1.version_number == 1
    assert db_version1.parameters_used == {"temp": 0.7}
    assert db_version1.fitness_score == 0.8

    version_data2 = PromptVersionCreate(content="Version 2 content")
    db_version2 = prompt_service_instance.create_prompt_version(db_session, prompt_id=prompt.id, version_create=version_data2)
    assert db_version2 is not None
    assert db_version2.version_number == 2

    # Test creating version for non-existent prompt
    version_for_non_existent_prompt = prompt_service_instance.create_prompt_version(db_session, prompt_id=9999, version_create=PromptVersionCreate(content="No prompt here"))
    assert version_for_non_existent_prompt is None


def test_get_prompt_version(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="PV Get Test"))
    created_version = prompt_service_instance.create_prompt_version(db_session, prompt.id, PromptVersionCreate(content="Content to get"))

    retrieved_version = prompt_service_instance.get_prompt_version(db_session, prompt_version_id=created_version.id)
    assert retrieved_version is not None
    assert retrieved_version.id == created_version.id
    assert retrieved_version.content == "Content to get"

    non_existent_version = prompt_service_instance.get_prompt_version(db_session, prompt_version_id=9999)
    assert non_existent_version is None

def test_get_prompt_versions_for_prompt(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt1 = prompt_service_instance.create_prompt(db_session, PromptCreate(name="PV List Test 1"))
    prompt2 = prompt_service_instance.create_prompt(db_session, PromptCreate(name="PV List Test 2"))

    v1_p1 = prompt_service_instance.create_prompt_version(db_session, prompt1.id, PromptVersionCreate(content="P1V1"))
    v2_p1 = prompt_service_instance.create_prompt_version(db_session, prompt1.id, PromptVersionCreate(content="P1V2"))
    v1_p2 = prompt_service_instance.create_prompt_version(db_session, prompt2.id, PromptVersionCreate(content="P2V1"))

    versions_p1 = prompt_service_instance.get_prompt_versions_for_prompt(db_session, prompt_id=prompt1.id)
    assert len(versions_p1) == 2
    assert {v.id for v in versions_p1} == {v1_p1.id, v2_p1.id}

    versions_p2 = prompt_service_instance.get_prompt_versions_for_prompt(db_session, prompt_id=prompt2.id)
    assert len(versions_p2) == 1
    assert versions_p2[0].id == v1_p2.id

    # Test pagination
    versions_p1_limit1 = prompt_service_instance.get_prompt_versions_for_prompt(db_session, prompt_id=prompt1.id, limit=1, skip=0)
    assert len(versions_p1_limit1) == 1
    # Assuming order by version_number (default or added in service)
    assert versions_p1_limit1[0].version_number == 1

    versions_p1_skip1 = prompt_service_instance.get_prompt_versions_for_prompt(db_session, prompt_id=prompt1.id, limit=1, skip=1)
    assert len(versions_p1_skip1) == 1
    assert versions_p1_skip1[0].version_number == 2


    versions_non_existent_prompt = prompt_service_instance.get_prompt_versions_for_prompt(db_session, prompt_id=9999)
    assert len(versions_non_existent_prompt) == 0

def test_update_prompt_version(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="PV Update Test"))
    db_version = prompt_service_instance.create_prompt_version(db_session, prompt.id, PromptVersionCreate(content="Old Content", parameters_used={"temp": 0.5}, fitness_score=0.7))

    update_data = PromptVersionUpdate(content="New Content", parameters_used={"temp": 0.8, "top_p": 0.9}, fitness_score=0.9)
    updated_version = prompt_service_instance.update_prompt_version(db_session, prompt_version_id=db_version.id, version_update=update_data)
    assert updated_version is not None
    assert updated_version.content == "New Content"
    assert updated_version.parameters_used == {"temp": 0.8, "top_p": 0.9}
    assert updated_version.fitness_score == 0.9

    update_partial = PromptVersionUpdate(content="Very New Content")
    updated_partial_version = prompt_service_instance.update_prompt_version(db_session, prompt_version_id=db_version.id, version_update=update_partial)
    assert updated_partial_version.content == "Very New Content"
    assert updated_partial_version.parameters_used == {"temp": 0.8, "top_p": 0.9} # Should persist
    assert updated_partial_version.fitness_score == 0.9 # Should persist

    update_non_existent = prompt_service_instance.update_prompt_version(db_session, prompt_version_id=9999, version_update=PromptVersionUpdate(content="No Such"))
    assert update_non_existent is None

def test_delete_prompt_version(db_session: SQLAlchemySession, prompt_service_instance: PromptService):
    prompt = prompt_service_instance.create_prompt(db_session, PromptCreate(name="PV Delete Test"))
    version_to_delete = prompt_service_instance.create_prompt_version(db_session, prompt.id, PromptVersionCreate(content="Delete Me Version"))

    # Create another version to ensure only one is deleted
    other_version = prompt_service_instance.create_prompt_version(db_session, prompt.id, PromptVersionCreate(content="Keep Me Version"))


    deleted_version = prompt_service_instance.delete_prompt_version(db_session, prompt_version_id=version_to_delete.id)
    assert deleted_version is not None
    assert deleted_version.id == version_to_delete.id
    assert prompt_service_instance.get_prompt_version(db_session, prompt_version_id=version_to_delete.id) is None

    # Ensure other version still exists
    assert prompt_service_instance.get_prompt_version(db_session, prompt_version_id=other_version.id) is not None

    # Ensure prompt still exists
    assert prompt_service_instance.get_prompt(db_session, prompt_id=prompt.id) is not None


    delete_non_existent = prompt_service_instance.delete_prompt_version(db_session, prompt_version_id=9999)
    assert delete_non_existent is None
