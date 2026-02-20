"""
Run only the data collection scheduler (no FastAPI server).
Use this if you want the scheduler in a separate process from the API.
Otherwise, the API server starts the scheduler automatically on startup.
"""
import sys
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.data_collection.scheduler import start_scheduler, stop_scheduler

def main():
    sched = start_scheduler()
    if sched is None:
        print("Scheduler is disabled (ENABLE_SCHEDULED_COLLECTION=False). Exiting.")
        return
    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        signal.pause()
    except AttributeError:
        import time
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    def _shutdown(signum, frame):
        stop_scheduler()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    main()
