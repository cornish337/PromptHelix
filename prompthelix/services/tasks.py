from celery import shared_task
import time # For a simple demonstration of async work

@shared_task
def example_task(message: str) -> str:
    # Simulate some work
    print(f"Received message for example_task: {message}")
    time.sleep(2) # Simulate a 2-second task
    result = f"Message processed: {message}"
    print(f"Finished processing for example_task: {message}")
    return result

@shared_task
def add(x: int, y: int) -> int:
    # Simulate some work
    print(f"Adding {x} + {y}")
    time.sleep(1) # Simulate a 1-second task
    result = x + y
    print(f"Result of add({x}, {y}) is {result}")
    return result

# To make these tasks discoverable, ensure celery_app.autodiscover_tasks()
# in prompthelix/celery_app.py is effective.
# When celery_app.autodiscover_tasks() is called (typically with no arguments
# or with a list of base packages like ['prompthelix']), Celery searches for
# modules named 'tasks.py' within the specified packages and their submodules.
# Given that prompthelix/celery_app.py calls autodiscover_tasks(), and this
# file is prompthelix/services/tasks.py, these tasks should be discovered
# automatically when the Celery worker starts, provided 'prompthelix' is
# in the Python path.
