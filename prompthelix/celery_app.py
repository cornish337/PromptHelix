import os
from celery import Celery

# Define Redis URL from environment variable, default to localhost
# Using different DB numbers for broker and backend is good practice.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", f"{REDIS_URL}/1")
CELERY_RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND_URL", f"{REDIS_URL}/2")

# Instantiate the Celery application
# The first argument is the name of the current module, used for generating task names.
# 'include' can be used to list modules to import when the worker starts.
celery_app = Celery(
    'prompthelix',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND_URL,
    include=[]  # Example: ['prompthelix.tasks', 'prompthelix.services.tasks']
)

# Celery Configuration Options
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',        # It's good practice to use UTC in your backend
    enable_utc=True,       # Ensures Celery uses UTC
    # Optional: Configure task execution settings if needed
    # task_acks_late = True,
    # worker_prefetch_multiplier = 1, # Might be useful for long-running tasks
    # result_expires=3600, # Time in seconds for results to be stored
)

# Autodiscover tasks: Celery will look for a tasks.py file in all installed Django apps
# or in packages listed in the 'include' argument of Celery().
# For a non-Django project, if your tasks are in 'tasks.py' files within your project
# structure that's part of Python's sys.path, this should work.
# Example: If you have prompthelix/services/tasks.py, it should be found
# if 'prompthelix' is in sys.path and 'services' is a package.
# Alternatively, you can explicitly list modules in the 'include' argument above.
celery_app.autodiscover_tasks()
# Example for more specific discovery if needed:
# celery_app.autodiscover_tasks(lambda: ['prompthelix.services', 'prompthelix.some_other_module'])

if __name__ == '__main__':
    # This script can be used to start a Celery worker.
    # Command: celery -A prompthelix.celery_app worker -l info
    # Or for development: celery -A prompthelix.celery_app worker -l debug
    celery_app.start()
