"""
GPU diagnostics and monitoring tools.
"""

import torch
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def check_availability():
    if torch.cuda.is_available():
        logging.info(f"CUDA: Yes | {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logging.info(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")
        return True
    else:
        logging.error("CUDA not available")
        return False


def test_computation():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    t1 = torch.randn(10000, 10000, device=device)
    t2 = torch.randn(10000, 10000, device=device)
    start = time.time()
    r = torch.mm(t1, t2)
    torch.cuda.synchronize()
    logging.info(f"Matrix multiply: {time.time() - start:.3f}s | shape={r.shape}")
    del t1, t2, r
    torch.cuda.empty_cache()


def memory_status():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    logging.info(f"GPU Mem: {alloc:.2f}/{total:.2f} GB (reserved: {reserved:.2f})")


def full_diagnostics():
    logging.info("=" * 70)
    logging.info("GPU DIAGNOSTICS")
    logging.info("=" * 70)
    check_availability()
    test_computation()
    memory_status()
