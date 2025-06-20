import redis
import os

# Default Redis connection parameters
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

redis_client = None # Initialize to None

try:
    # Attempt to create a Redis client instance
    temp_redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True # To get strings back from Redis, not bytes
    )
    # Test connection
    temp_redis_client.ping()
    redis_client = temp_redis_client # Assign to global variable if successful
    print("Successfully connected to Redis.") # Or use logging
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}") # Or use logging
    # redis_client remains None as initialized
    # Depending on application needs, could raise error or handle elsewhere
except Exception as e: # Catch other potential errors during Redis client init
    print(f"An unexpected error occurred during Redis client initialization: {e}")
    # redis_client remains None

# To make it easily importable:
# from prompthelix.redis_client import redis_client

# Example of how to use it (optional, for direct script execution):
if __name__ == "__main__":
    if redis_client:
        print("Redis client is available.")
        # You can try setting and getting a key here for further testing
        # redis_client.set("mykey", "Hello Redis")
        # value = redis_client.get("mykey")
        # print(f"Got value: {value}")
    else:
        print("Redis client is not available. Check connection settings or Redis server.")
