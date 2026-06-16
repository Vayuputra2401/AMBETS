#!/bin/bash
# GCP GPU Instance Startup Script for NeuroReAct Training
# This script runs automatically when the instance starts

set -e  # Exit on any error

echo "=== NeuroReAct GPU Instance Startup ==="
echo "Starting at: $(date)"

# Update system
echo "Updating system packages..."
sudo apt-get update -y

# Install essential tools
echo "Installing essential tools..."
sudo apt-get install -y tmux htop git wget curl unzip

# Verify GPU setup
echo "Verifying GPU setup..."
nvidia-smi
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install ultralytics
pip3 install nibabel
pip3 install scikit-learn
pip3 install opencv-python
pip3 install matplotlib
pip3 install tqdm
pip3 install psutil
pip3 install pyyaml

# Create working directory
mkdir -p /home/$(whoami)/neuroreact
cd /home/$(whoami)/neuroreact

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/$(whoami)/neuroreact:$PYTHONPATH

echo "=== Environment Setup Complete ==="
echo "Ready for training at: $(date)"

# Optional: Auto-start training if code and data are present
if [ -f "/home/$(whoami)/detector_agent/train/train.py" ] && [ -d "/home/$(whoami)/brats_yolo_dataset" ]; then
    echo "=== Auto-starting training ==="
    cd /home/$(whoami)/detector_agent/train
    
    # Update dataset path in config if needed
    sed -i 's|D:/brats_yolo_dataset|/home/'$(whoami)'/brats_yolo_dataset|g' training_config.yaml
    
    # Start training in tmux session
    tmux new-session -d -s neuroreact-training 'cd /home/'$(whoami)'/detector_agent/train && python3 train.py'
    echo "Training started in tmux session 'neuroreact-training'"
    echo "Use 'tmux attach -t neuroreact-training' to monitor"
    
    # Optional: Auto-shutdown after training (uncomment if desired)
    # echo "Setting up auto-shutdown after training completion..."
    # (tmux capture-pane -t neuroreact-training -p | grep -q "Training completed" && sudo shutdown -h +5) &
fi

echo "=== Startup script completed at: $(date) ===" 