"""
GPU Setup Checker for Deep Learning Training
"""

import sys

def check_gpu_setup():
    print("="*60)
    print("GPU SETUP CHECKER")
    print("="*60)
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            
            # Get GPU memory info
            device = torch.cuda.get_device_properties(0)
            total_memory = device.total_memory / 1024**3
            print(f"✓ GPU memory: {total_memory:.1f} GB")
            
            # Check current memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"✓ Memory allocated: {allocated:.2f} GB")
            print(f"✓ Memory cached: {cached:.2f} GB")
            print(f"✓ Memory available: {total_memory - cached:.1f} GB")
            
            # Test basic GPU operations
            print("\nTesting GPU operations...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                print("✓ GPU tensor operations working")
                
                # Test mixed precision
                from torch.cuda.amp import autocast
                with autocast():
                    z_fp16 = torch.mm(x, y)
                print("✓ Mixed precision (FP16) working")
                
                del x, y, z, z_fp16  # Clean up memory
                torch.cuda.empty_cache()
                print("✓ GPU memory cleanup working")
                
            except Exception as e:
                print(f"✗ GPU operations failed: {e}")
            
        else:
            print("✗ CUDA not available - GPU training not possible")
            print("  Possible solutions:")
            print("  1. Install CUDA-enabled PyTorch")
            print("  2. Check NVIDIA drivers")
            print("  3. Verify CUDA installation")
    
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Check other common libraries
    print("\n" + "="*60)
    print("DEEP LEARNING LIBRARIES")
    print("="*60)
    
    libraries = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('nibabel', 'NiBabel (for medical imaging)'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'Progress bars'),
        ('tensorboard', 'TensorBoard (optional)'),
        ('cv2', 'OpenCV (optional)')
    ]
    
    for lib_name, description in libraries:
        try:
            __import__(lib_name)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} - not installed")
    
    # Performance test
    if 'torch' in sys.modules and torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU PERFORMANCE TEST")
        print("="*60)
        
        try:
            import time
            
            # Test different tensor sizes to estimate capacity
            test_sizes = [
                (512, 512, 64),    # Small 3D patch
                (1024, 1024, 32),  # Medium 2D batch
                (2048, 2048, 16),  # Large 2D batch
            ]
            
            for h, w, d in test_sizes:
                try:
                    start_time = time.time()
                    x = torch.randn(1, 1, h, w, d).cuda()  # Batch of 1
                    
                    # Simple convolution test
                    conv = torch.nn.Conv3d(1, 32, kernel_size=3, padding=1).cuda()
                    y = conv(x)
                    
                    end_time = time.time()
                    
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"✓ {h}×{w}×{d} tensor: {end_time-start_time:.3f}s, {memory_used:.2f}GB")
                    
                    del x, y, conv
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"✗ {h}×{w}×{d} tensor: Out of memory")
                    else:
                        print(f"✗ {h}×{w}×{d} tensor: {e}")
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Performance test failed: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR RTX 2050")
    print("="*60)
    
    print("Your RTX 2050 with 4GB VRAM can handle:")
    print("✓ Small to medium neural networks")
    print("✓ 2D medical image segmentation")
    print("✓ Transfer learning with pre-trained models")
    print("✓ 3D patches (64×64×64 to 96×96×96)")
    print()
    print("Optimal settings for BraTS training:")
    print("• Batch size: 1-2 for 3D data, 4-8 for 2D data")
    print("• Use mixed precision: torch.cuda.amp")
    print("• Patch size: 64×64×64 or 96×96×96 for 3D")
    print("• Image size: 224×224 or 256×256 for 2D")
    print("• Enable gradient checkpointing for larger models")
    print("• Use gradient accumulation for effective larger batches")

if __name__ == "__main__":
    check_gpu_setup() 