"""
simple_esgran_upscale_cpu.py
----------------------------
Minimal Real-ESRGAN script that runs exclusively on CPU.
Optimizes for available CPU instructions (MKL, AVX, OpenMP).
Does nothing beyond upscaling an image on CPU.

Author: Shawn
"""

import torch
import cv2
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def check_cpu_optimizations():
    """ Check available CPU optimizations and enable them if supported. """
    optimizations = {
        "MKL": torch._C.has_mkl,
        "MKL-DNN": torch.backends.mkldnn.enabled,
        "OpenMP": torch.backends.openmp.is_available(),
        #"AVX": torch.has_avx,
        #"AVX2": torch.has_avx2,
        #"FMA": torch.has_fma,
    }
    
    print("\n🔍 CPU Optimizations Available:")
    for opt, available in optimizations.items():
        print(f"  {opt}: {'✅ Enabled' if available else '❌ Not Available'}")

    if not any(optimizations.values()):
        print("⚠ No CPU optimizations detected. Performance may be slow.")

def load_model(model_path, scale):
    """ Load Real-ESRGAN model, forcing CPU execution """
    #This line creates an instance of the RRDBNet model, which is a ResNet-based super-resolution network used in Real-ESRGAN.
    """
        ### **📌 Breaking Down Each Parameter**
        | Parameter        | Value  | What It Does |
        |------------------|--------|-----------------------------------------------------------------------------------------|
        | `num_in_ch=3`    | `3`    | Number of input channels (3 for **RGB images**).                                        |
        | `num_out_ch=3`   | `3`    | Number of output channels (**also RGB**).                                               |
        | `num_feat=64`    | `64`   | Number of feature maps in the first convolution layer (controls model size).            |
        | `num_block=23`   | `23`   | Number of **Residual-in-Residual Dense Blocks (RRDB)** in the network (controls depth). |
        | `num_grow_ch=32` | `32`   | Growth factor for feature channels in dense connections (controls learning power).      |
        | `scale=scale`    | `4` (or user-defined) | Upscaling factor (e.g., **4x for 4x resolution boost**).                 |
    """    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    device = "cpu"
    
    upscaler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=200,  # Adjust if needed
        tile_pad=10,
        pre_pad=0,
        half=False,  # Disable FP16 (not needed for CPU)
        device=device
    )

    return upscaler

def upscale_image(input_path, output_path, scale=4):
    """ Perform image upscaling on CPU """
    model_path = "RealESRGAN_x4.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found. Download it from the official repo.")
        return

    check_cpu_optimizations()
    upscaler = load_model(model_path, scale)

    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Failed to load image: {input_path}")
        return

    #Converts an image from OpenCV's default BGR color format to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Enhance resolution
    output, _extra = upscaler.enhance(img, outscale=scale)
    meta = extra[0] if extra else None
    # Save output
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)
    print(f"✅ Upscaled image saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-ESRGAN CPU-only Image Upscaler")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save upscaled image")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4)")
    args = parser.parse_args()

    upscale_image(args.input, args.output, scale=args.scale)
