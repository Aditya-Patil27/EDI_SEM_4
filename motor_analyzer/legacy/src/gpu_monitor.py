"""
Real-time GPU monitoring during training
Run this in a separate terminal while training is running:
  python src/gpu_monitor.py
"""

import torch
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def monitor_gpu(interval=2, duration=None):
    """
    Monitor GPU usage in real-time.
    
    Args:
        interval: Update interval in seconds
        duration: Total monitoring duration (None = infinite)
    """
    
    if not torch.cuda.is_available():
        logging.error("GPU not available!")
        return
    
    device = torch.device('cuda')
    logging.info(f"Monitoring GPU: {torch.cuda.get_device_name(device)}")
    logging.info(f"Update interval: {interval}s")
    logging.info("Press Ctrl+C to stop\n")
    
    # Print header
    print(f"\n{'Time':<10} {'Memory':<15} {'Util%':<10} {'Allocated':<15} {'Reserved':<15}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break
            
            try:
                torch.cuda.synchronize()
                
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total = torch.cuda.get_device_properties(device).total_memory
                
                allocated_gb = allocated / 1e9
                reserved_gb = reserved / 1e9
                total_gb = total / 1e9
                
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                current_time = datetime.now().strftime("%H:%M:%S")
                memory_str = f"{allocated_gb:.2f}/{total_gb:.2f}GB"
                
                print(f"{current_time:<10} {memory_str:<15} {utilization:>7.1f}% {allocated_gb:>10.2f}GB {reserved_gb:>14.2f}GB")
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error reading GPU stats: {e}")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        logging.info("Final GPU memory status:")
        check_gpu_memory_summary()

def check_gpu_memory_summary():
    """Print summary of GPU memory."""
    if not torch.cuda.is_available():
        return
    
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    
    logging.info(f"  Allocated: {allocated:.2f} GB")
    logging.info(f"  Reserved: {reserved:.2f} GB")
    logging.info(f"  Peak: {peak:.2f} GB")
    logging.info(f"  Total: {total:.2f} GB")
    logging.info(f"  Available: {total - allocated:.2f} GB")

if __name__ == "__main__":
    import sys
    
    interval = 2
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            logging.warning(f"Invalid interval '{sys.argv[1]}', using default 2s")
    
    monitor_gpu(interval=interval)
