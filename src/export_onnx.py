import os
import argparse
import torch
import torch.onnx

from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING

def main():
    parser = argparse.ArgumentParser(description="Compile PyTorch Whale SED Model to ONNX")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to trained PyTorch weights")
    parser.add_argument("--onnx_path", type=str, default="models/best_model.onnx", help="Path to save output ONNX model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
        
    print(f"Loading PyTorch model from: {args.model_path}")
    
    # Load on CPU for dynamic cross-platform deployment compatibility
    device = torch.device("cpu")
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()
    
    # Create dummy input representative of one normalized spectrogram clip
    # Shape: (batch_size=1, channels=1, freq_bins=129, time_frames=118)
    dummy_input = torch.randn(1, 1, 129, 118, device=device)
    
    print(f"Exporting model to ONNX format at: {args.onnx_path}...")
    
    # Ensure intermediate directories exist
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    
    # Export graph
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=14,                  # Standard opset version with robust operator coverage
        do_constant_folding=True,          # Optimize constants
        input_names=["input_spectrogram"],  # Input name
        output_names=["output_probs"],      # Output name
        # Allow dynamic batch sizes (e.g. for variable sliding-window batching)
        dynamic_axes={
            "input_spectrogram": {0: "batch_size"},
            "output_probs": {0: "batch_size"}
        }
    )
    
    print("ONNX model compilation completed successfully!")
    print(f"Compiled model size: {os.path.getsize(args.onnx_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
