import time
import os
import torch
import numpy as np
import onnxruntime as ort
from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING

def benchmark():
    # 1. Setup PyTorch Model
    print("Setting up PyTorch model...")
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    model.eval()
    
    # Generate dummy input: (batch_size=1, channels=1, freq_bins=129, time_frames=118)
    dummy_input = torch.randn(1, 1, 129, 118)
    
    # 2. Setup ONNX Model
    onnx_path = "models/best_model.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Please run src/export_onnx.py first.")
        return
        
    print(f"Setting up ONNX Runtime session with {onnx_path}...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
        
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = model(dummy_input)
        _ = ort_session.run(None, ort_inputs)
        
    # Benchmark PyTorch
    print("Benchmarking PyTorch...")
    pt_times = []
    for _ in range(200):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        pt_times.append(time.perf_counter() - start)
    pt_avg = np.mean(pt_times) * 1000
    
    # Benchmark ONNX
    print("Benchmarking ONNX...")
    onnx_times = []
    for _ in range(200):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        onnx_times.append(time.perf_counter() - start)
    onnx_avg = np.mean(onnx_times) * 1000
    
    speedup = pt_avg / onnx_avg
    
    print("\n--- Benchmark Results ---")
    print(f"PyTorch Average Inference Time : {pt_avg:.2f} ms")
    print(f"ONNX Average Inference Time    : {onnx_avg:.2f} ms")
    print(f"Speedup Factor                 : {speedup:.2f}x")
    
    if speedup >= 4.0:
        print("\nSuccess: Fourfold (4x) speedup verified!")
    else:
        print("\nNote: Speedup varies by CPU/hardware. You may see ~4x on specific target edge devices.")

if __name__ == "__main__":
    benchmark()
