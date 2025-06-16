import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session as DBSession
from sqlalchemy import select

# Assuming prompthelix is in PYTHONPATH or current directory structure
from prompthelix.database import SessionLocal, engine
from prompthelix.models.base import Base
from prompthelix.models.user_models import User
from prompthelix.models.prompt_models import Prompt, PromptVersion

def main():
    print("Starting prompt owner_id test script...")

    # Create tables if they don't exist (Alembic should handle this, but good for standalone script)
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables checked/created.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise

    db: DBSession = SessionLocal()
    print("Database session started.")

    test_user = None
    new_prompt_id_for_test = None

    try:
        # 1. Create or get a test user
        test_username = "testuser"
        stmt = select(User).where(User.username == test_username)
        test_user = db.execute(stmt).scalar_one_or_none()

        if test_user:
            print(f"User '{test_username}' found with ID: {test_user.id}")
        else:
            print(f"User '{test_username}' not found. Creating new user...")
            test_user = User(
                username=test_username,
                email="testuser@example.com",
                hashed_password="a_very_secure_password" # In a real app, hash this properly
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            print(f"User '{test_username}' created with ID: {test_user.id}")

        # 2. Attempt to create a new Prompt record with owner_id
        print("Creating new Prompt and PromptVersion...")
        new_prompt = Prompt(
            name="Test Prompt by testuser",
            description="A prompt to test owner_id functionality",
            owner_id=test_user.id
        )
        db.add(new_prompt)
        db.commit()
        db.refresh(new_prompt)
        new_prompt_id_for_test = new_prompt.id # Store for later specific check
        print(f"Prompt created with ID: {new_prompt.id}, Owner ID: {new_prompt.owner_id}")

        assert new_prompt.owner_id == test_user.id, \
            f"Created prompt's owner_id {new_prompt.owner_id} does not match user's ID {test_user.id}"

        new_prompt_version = PromptVersion(
            prompt_id=new_prompt.id,
            version_number=1,
            content="This is the initial version of the test prompt."
            # parameters_used can be {} or None
        )
        db.add(new_prompt_version)
        db.commit()
        db.refresh(new_prompt_version)
        print(f"PromptVersion created with ID: {new_prompt_version.id} for Prompt ID: {new_prompt.id}")

        # 3. Attempt to fetch prompts from the database
        print("Fetching all prompts to verify owner_id presence...")
        all_prompts = db.query(Prompt).all()

        if not all_prompts:
            raise AssertionError("No prompts found in the database after creation.")

        print(f"Found {len(all_prompts)} prompts.")

        # 4. Verify that the fetched prompts include the owner_id
        found_our_prompt = False
        for p in all_prompts:
            print(f"  Prompt ID: {p.id}, Name: {p.name}, Owner ID: {p.owner_id}")
            if not hasattr(p, 'owner_id'):
                raise AttributeError(f"Prompt ID {p.id} is missing 'owner_id' attribute.")
            if p.owner_id is None: # Should not happen due to nullable=False
                 raise ValueError(f"Prompt ID {p.id} has None for owner_id.")
            if p.id == new_prompt_id_for_test:
                found_our_prompt = True
                assert p.owner_id == test_user.id, \
                    f"Fetched prompt ID {p.id} has owner_id {p.owner_id}, expected {test_user.id}"
                print(f"Verified owner_id for the newly created prompt (ID: {p.id}).")

        if not found_our_prompt and new_prompt_id_for_test is not None:
            raise AssertionError(f"The newly created prompt (ID: {new_prompt_id_for_test}) was not found in the all_prompts query.")

        print("Successfully fetched prompts and verified owner_id presence and correctness.")
        print("Test script completed successfully!")

    except AssertionError as ae:
        print(f"Assertion Error: {ae}")
        raise
    except AttributeError as ate:
        print(f"Attribute Error: {ate}") # e.g. "no such column" would manifest here
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if db:
            db.close()
            print("Database session closed.")

if __name__ == "__main__":
    main()
