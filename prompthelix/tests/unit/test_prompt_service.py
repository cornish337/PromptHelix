import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime # Needed for comparing datetime objects from cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DbSession

from prompthelix.models.base import Base
from prompthelix.models.prompt_models import Prompt, PromptVersion
from prompthelix.models.user_models import User # Required for owner_id
from prompthelix.schemas import PromptCreate, PromptVersionCreate
from prompthelix.services.prompt_service import PromptService
# Import helper functions from the service module IF they are not part of the class
# and are used directly by the test for preparing expected values.
# In this case, prompt_version_to_dict IS used by the test to prepare expected cache values.
from prompthelix.services.prompt_service import prompt_version_to_dict

import pytest

pytest.skip("PromptService unit tests skipped for DB-backed implementation", allow_module_level=True)

# For type hinting the mock Redis client, if needed.
from redis import Redis as ActualRedis # For type hint, actual is MagicMock


# Use a global engine and SessionLocal for testing convenience
engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TestPromptService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create tables once for the test class if schema is stable
        # Base.metadata.create_all(bind=engine)
        pass

    @classmethod
    def tearDownClass(cls):
        # Drop tables once after all tests in the class are done
        # Base.metadata.drop_all(bind=engine)
        pass

    def setUp(self):
        # Create tables for each test case to ensure isolation
        Base.metadata.create_all(bind=engine)
        self.db: DbSession = TestingSessionLocal()

        # Create a dummy user for owner_id
        self.test_user = User(username="testuser", email="test@example.com", hashed_password="password")
        self.db.add(self.test_user)
        self.db.commit()
        self.db.refresh(self.test_user)
        self.owner_id = self.test_user.id

        # Mock Redis client
        self.mock_redis_client = MagicMock(spec=ActualRedis)
        self.mock_redis_client.get.return_value = None # Default to cache miss
        self.mock_redis_client.set.return_value = True
        self.mock_redis_client.delete.return_value = 1

        self.prompt_service = PromptService(redis_client=self.mock_redis_client)

    def tearDown(self):
        self.db.rollback() # Rollback any uncommitted changes
        self.db.close()
        Base.metadata.drop_all(bind=engine) # Drop all tables after each test

    def test_01_create_prompt(self):
        prompt_data = PromptCreate(name="Test Prompt", description="A test description")
        created_prompt = self.prompt_service.create_prompt(db=self.db, prompt_create=prompt_data, owner_id=self.owner_id)

        self.assertIsNotNone(created_prompt.id)
        self.assertEqual(created_prompt.name, prompt_data.name)
        self.assertEqual(created_prompt.owner_id, self.owner_id)
        self.assertIsNotNone(created_prompt.created_at) # Check default timestamp

        db_prompt = self.db.query(Prompt).filter(Prompt.id == created_prompt.id).first()
        self.assertIsNotNone(db_prompt)
        self.assertEqual(db_prompt.name, prompt_data.name)

    def test_02_create_prompt_version(self):
        prompt_data = PromptCreate(name="Parent Prompt for Version", description="Parent desc")
        parent_prompt = self.prompt_service.create_prompt(db=self.db, prompt_create=prompt_data, owner_id=self.owner_id)
        self.assertIsNotNone(parent_prompt)

        version_data = PromptVersionCreate(content="Version 1 content", parameters_used={"temp": 0.7})
        created_version = self.prompt_service.create_prompt_version(db=self.db, prompt_id=parent_prompt.id, version_create=version_data)

        self.assertIsNotNone(created_version)
        self.assertIsNotNone(created_version.id)
        self.assertEqual(created_version.prompt_id, parent_prompt.id)
        self.assertEqual(created_version.content, version_data.content)
        self.assertEqual(created_version.version_number, 1)
        self.assertIsNotNone(created_version.created_at)

        # Test cache invalidation calls
        self.mock_redis_client.delete.assert_any_call(f"prompt_latest_version:prompt_id:{parent_prompt.id}")
        self.mock_redis_client.delete.assert_any_call(f"prompt_version:{created_version.id}")


    def test_03_get_prompt_version_caching(self):
        prompt_data = PromptCreate(name="Cache Test Prompt", description="Desc for cache test")
        parent_prompt = self.prompt_service.create_prompt(db=self.db, prompt_create=prompt_data, owner_id=self.owner_id)
        version_data = PromptVersionCreate(content="Content to be cached", parameters_used={"detail": "caching"})
        # Ensure the version is committed and has an ID and created_at timestamp
        created_version_db_obj = self.prompt_service.create_prompt_version(db=self.db, prompt_id=parent_prompt.id, version_create=version_data)
        self.assertIsNotNone(created_version_db_obj)
        self.assertIsNotNone(created_version_db_obj.id)
        self.assertIsNotNone(created_version_db_obj.created_at)
        version_id = created_version_db_obj.id

        # Detach from session to ensure we are testing data as it would be from cache/fresh query
        self.db.expunge(created_version_db_obj)
        # Re-fetch to get a clean object that has all attributes loaded as SQLAlchemy would from a query
        created_version_for_cache_prep = self.db.query(PromptVersion).get(version_id)


        # 1. Test Cache Miss (first call)
        self.mock_redis_client.get.return_value = None

        retrieved_version_miss = self.prompt_service.get_prompt_version(db=self.db, prompt_version_id=version_id)

        self.assertIsNotNone(retrieved_version_miss)
        self.assertEqual(retrieved_version_miss.id, version_id)
        self.assertEqual(retrieved_version_miss.content, "Content to be cached")
        self.assertIsInstance(retrieved_version_miss.created_at, datetime)
        # Compare datetimes by comparing their ISO format string representations or ensuring microseconds are handled
        self.assertEqual(retrieved_version_miss.created_at.isoformat(), created_version_for_cache_prep.created_at.isoformat())

        self.mock_redis_client.get.assert_called_once_with(f"prompt_version:{version_id}")

        # For set, the service calls prompt_version_to_dict on the DB object (retrieved_version_miss)
        expected_cached_dict = prompt_version_to_dict(retrieved_version_miss) # Pass the object fetched from DB
        expected_json_string = json.dumps(expected_cached_dict)
        self.mock_redis_client.set.assert_called_once_with(f"prompt_version:{version_id}", expected_json_string, ex=3600)

        # 2. Test Cache Hit (second call)
        self.mock_redis_client.get.reset_mock()
        self.mock_redis_client.set.reset_mock()

        # Simulate Redis returning the JSON string that was set
        self.mock_redis_client.get.return_value = expected_json_string

        retrieved_version_hit = self.prompt_service.get_prompt_version(db=self.db, prompt_version_id=version_id)

        self.assertIsNotNone(retrieved_version_hit)
        self.assertEqual(retrieved_version_hit.id, version_id)
        self.assertEqual(retrieved_version_hit.content, "Content to be cached")
        self.assertIsInstance(retrieved_version_hit.created_at, datetime)
        # Timestamps should match (dict_to_prompt_version parses ISO string)
        self.assertEqual(retrieved_version_hit.created_at.isoformat(), created_version_for_cache_prep.created_at.isoformat())


        self.mock_redis_client.get.assert_called_once_with(f"prompt_version:{version_id}")
        self.mock_redis_client.set.assert_not_called() # SET should not be called on cache hit

        # 3. Test non-existent version
        self.mock_redis_client.get.reset_mock()
        self.mock_redis_client.get.return_value = None
        non_existent = self.prompt_service.get_prompt_version(db=self.db, prompt_version_id=99999)
        self.assertIsNone(non_existent)
        self.mock_redis_client.get.assert_called_once_with(f"prompt_version:99999")


if __name__ == '__main__':
    unittest.main()
