"""
GPU Diagnostics and Monitoring Tool
"""

import torch
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def check_gpu_availability():
    """Check if GPU is available and print detailed info."""
    logging.info("=" * 70)
    logging.info("GPU AVAILABILITY CHECK")
    logging.info("=" * 70)
    
    if torch.cuda.is_available():
        logging.info(f"✓ CUDA available: Yes")
        try:
            cuda_version = torch.version.cuda
        except AttributeError:
            cuda_version = "Unknown"
        logging.info(f"✓ CUDA version: {cuda_version}")
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = "Unknown"
        logging.info(f"✓ cuDNN version: {cudnn_version}")
        logging.info(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1e9
            logging.info(f"\n  GPU {i}: {props.name}")
            logging.info(f"    Memory: {memory_total:.1f} GB")
            logging.info(f"    Compute Capability: {props.major}.{props.minor}")
        
        logging.info(f"\n✓ Current Device: {torch.cuda.current_device()}")
        logging.info(f"✓ Current Device Name: {torch.cuda.get_device_name()}")
        
        return True
    else:
        logging.error("✗ CUDA available: No - GPU not available!")
        logging.error("  Make sure NVIDIA GPU drivers and CUDA toolkit are installed.")
        return False

def test_gpu_computation():
    """Test actual GPU computation with a simple operation."""
    logging.info("\n" + "=" * 70)
    logging.info("GPU COMPUTATION TEST")
    logging.info("=" * 70)
    
    if not torch.cuda.is_available():
        logging.error("Cannot test: GPU not available.")
        return False
    
    try:
        # Create a large tensor and test computation
        device = torch.device('cuda')
        size = 10000
        
        logging.info(f"Creating {size}x{size} tensor on GPU...")
        t1 = torch.randn(size, size, device=device)
        t2 = torch.randn(size, size, device=device)
        
        logging.info("Running matrix multiplication on GPU...")
        start = time.time()
        result = torch.mm(t1, t2)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed = time.time() - start
        
        logging.info(f"✓ Matrix multiplication took {elapsed:.3f} seconds")
        logging.info(f"✓ Result shape: {result.shape}")
        
        # Clear GPU memory
        del t1, t2, result
        torch.cuda.empty_cache()
        
        logging.info("✓ GPU computation test passed!")
        return True
    except Exception as e:
        logging.error(f"✗ GPU computation test failed: {e}")
        return False

def check_gpu_memory():
    """Check current GPU memory usage."""
    logging.info("\n" + "=" * 70)
    logging.info("GPU MEMORY STATUS")
    logging.info("=" * 70)
    
    if not torch.cuda.is_available():
        logging.error("Cannot check: GPU not available.")
        return
    
    device = torch.device('cuda')
    
    torch.cuda.reset_peak_memory_stats()
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    logging.info(f"Allocated: {allocated:.2f} GB")
    logging.info(f"Reserved: {reserved:.2f} GB")
    logging.info(f"Total: {total:.2f} GB")
    logging.info(f"Available: {total - allocated:.2f} GB")
    
    utilization = (allocated / total) * 100
    logging.info(f"GPU Utilization: {utilization:.1f}%")

def test_dataloader_speed():
    """Test DataLoader speed with GPU tensors."""
    logging.info("=" * 70)
    logging.info("DATALOADER SPEED TEST")
    logging.info("=" * 70)
    
    if not torch.cuda.is_available():
        logging.error("Cannot test: GPU not available.")
        return
    
    device = torch.device('cuda')
    
    try:
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        X = torch.randn(1000, 1, 60)  # 60 features like your model
        y = torch.randint(0, 3, (1000,))
        dataset = TensorDataset(X, y)
        
        batch_size = 128
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, pin_memory=True, drop_last=True)
        
        logging.info(f"Dataset size: {len(dataset)}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Number of batches: {len(loader)}")
        
        # Time 5 epochs
        total_time = 0
        num_batches = 0
        
        logging.info("Running 5 iterations through DataLoader...")
        for epoch in range(5):
            start = time.time()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                torch.cuda.synchronize()
                num_batches += 1
            
            elapsed = time.time() - start
            total_time += elapsed
            logging.info(f"  Epoch {epoch+1}: {elapsed:.2f}s ({len(loader) / elapsed:.1f} batches/sec)")
        
        avg_time_per_batch = total_time / num_batches
        logging.info(f"\nAverage time per batch: {avg_time_per_batch*1000:.2f} ms")
        logging.info(f"Throughput: {1/avg_time_per_batch:.1f} batches/sec")
        logging.info("✓ DataLoader test completed")
        
    except Exception as e:
        logging.error(f"✗ DataLoader test failed: {e}")

def diagnose_gpu_issues():
    """Run full GPU diagnostics."""
    logging.info("\n" + "=" * 70)
    logging.info("COMPLETE GPU DIAGNOSTICS")
    logging.info("=" * 70)
    
    checks = [
        ("GPU Availability", check_gpu_availability),
        ("GPU Computation", test_gpu_computation),
        ("GPU Memory", check_gpu_memory),
        ("DataLoader Speed", test_dataloader_speed),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            if callable(check_func) and check_func.__code__.co_argcount == 0:
                result = check_func()
                if isinstance(result, bool):
                    results.append((name, "PASS" if result else "FAIL"))
            else:
                check_func()
                results.append((name, "PASS"))
        except Exception as e:
            logging.error(f"Error in {name}: {e}")
            results.append((name, "ERROR"))
    
    logging.info("=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        logging.info(f"{symbol} {name}: {status}")
    
    logging.info("=" * 70)
    logging.info("RECOMMENDATIONS")
    logging.info("=" * 70)
    
    if check_gpu_availability():
        logging.info("✓ GPU is available and ready to use")
        logging.info("✓ Ensure training script uses: device = torch.device('cuda')")
        logging.info("✓ Move model to GPU: model.to(device)")
        logging.info("✓ Move batches to GPU in loop: batch_X.to(device, non_blocking=True)")
        logging.info("✓ Use pin_memory=True in DataLoader")
        logging.info("✓ Use batch_size >= 128 for better GPU utilization")
    else:
        logging.error("✗ GPU not available - check NVIDIA driver and CUDA installation")
        logging.error("  Run: nvidia-smi")

if __name__ == "__main__":
    diagnose_gpu_issues()
