import sys
import os

# Ensure the server directory is in the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "server"))

from server.inference import run_benchmark, run_task, main

if __name__ == "__main__":
    main()
