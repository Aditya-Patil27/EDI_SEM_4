"""
Real-time GPU memory monitor. Run in separate terminal:
  python -m utils.gpu_monitor
"""

import torch
import time
from datetime import datetime


def monitor(interval=2, duration=None):
    if not torch.cuda.is_available():
        print("GPU not available")
        return

    device = torch.device("cuda")
    print(f"Monitoring: {torch.cuda.get_device_name(device)} ({interval}s interval)")
    print(f"{'Time':<10} {'Allocated':<15} {'Reserved':<15} {'Total':<10}")
    print("-" * 50)

    start = time.time()
    try:
        while True:
            if duration and time.time() - start > duration:
                break
            t = datetime.now().strftime("%H:%M:%S")
            alloc = torch.cuda.memory_allocated(device) / 1e9
            res = torch.cuda.memory_reserved(device) / 1e9
            total = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"{t:<10} {alloc:<10.2f}GB {res:<10.2f}GB {total:<10.1f}GB")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    monitor(interval=interval)
