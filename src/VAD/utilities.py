import time
from functools import wraps


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Convert elapsed time to HH:MM:SS format
        h = int(elapsed_time // 3600)
        m = int((elapsed_time % 3600) // 60)
        s = int(elapsed_time % 60)

        # Print the elapsed time
        print(f"Time spent: {h}h{m:02}min{s:02}s")

    return wrapper
