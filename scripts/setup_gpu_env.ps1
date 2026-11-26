# Match PRISM-style GPU training: Python 3.11 + CUDA PyTorch (same stack that works on RTX 3050).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -U pip
# CUDA 12.1 wheel (aligned with existing Python 3.11 install: torch 2.5.1+cu121)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

$env:PYTHONPATH = "src"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"