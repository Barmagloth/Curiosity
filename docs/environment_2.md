# Environment Setup — PC 2 (Nvidia)

## Hardware

- **GPU**: NVIDIA GeForce RTX 2070 (Turing, compute capability 7.5, 8 GB VRAM)
- **OS**: Windows 10 Enterprise 10.0.19045
- **Driver**: 581.80 (supports CUDA up to 13.0)

## GPU virtual environment (.venv-gpu)

- **Python**: 3.12.11
- **Virtual environment**: `.venv-gpu` in project root
- **PyTorch**: 2.10.0+cu128
- **Backend**: CUDA 12.8
- **GPU detected**: NVIDIA GeForce RTX 2070

### Replication instructions

```bash
cd R:\Projects\Curiosity

# Python 3.12 source (adjust path if needed):
"E:\webui2\Output\Assets\Python\cpython-3.12.11-windows-x86_64-none\python.exe" -m venv .venv-gpu

.venv-gpu\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy matplotlib
```

### Verification

```bash
.venv-gpu\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); t = torch.rand(3,3, device='cuda'); print(t)"
# Expected: PyTorch: 2.10.0+cu128, CUDA: True, GPU: NVIDIA GeForce RTX 2070
```

### CUDA benchmark results (2026-03-18)

Matrix multiply, CPU vs CUDA on RTX 2070:

| Size | CPU | CUDA | Speedup |
|------|-----|------|---------|
| 512x512 | 0.9 ms | 0.09 ms | 9.8x |
| 1024x1024 | 6.4 ms | 0.55 ms | 11.6x |
| 2048x2048 | 48.2 ms | 3.36 ms | 14.4x |
| 4096x4096 | 340.4 ms | 19.75 ms | 17.2x |

CUDA shows strong speedup at all sizes. Compared to PC 1 (AMD Radeon 780M + DirectML, 1.8x at 2048), this setup is ~7x faster on GPU at the same size.

### Installed packages — GPU venv (pip freeze)

```
contourpy==1.3.3
cycler==0.12.1
filelock==3.20.0
fonttools==4.62.1
fsspec==2025.12.0
Jinja2==3.1.6
kiwisolver==1.5.0
MarkupSafe==3.0.2
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.3.5
packaging==26.0
pillow==12.0.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy==1.17.1
setuptools==70.2.0
six==1.17.0
sympy==1.14.0
torch==2.10.0+cu128
torchvision==0.25.0+cu128
typing_extensions==4.15.0
```

## Differences from PC 1 (environment_1.md)

| | PC 1 (AMD) | PC 2 (Nvidia) |
|---|---|---|
| GPU | Radeon 780M (iGPU, RDNA 3) | RTX 2070 (discrete, Turing) |
| VRAM | Shared | 8 GB dedicated |
| Backend | DirectML (`torch-directml`) | CUDA 12.8 (native) |
| PyTorch | 2.4.1 (pinned by directml) | 2.10.0+cu128 |
| Python | 3.12.10 | 3.12.11 |
| Speedup @ 2048 | 1.8x | 14.4x |

### Code compatibility note

Both environments use the same Python 3.12 + PyTorch API. Code written for PC 1 should work on PC 2 without changes — the only difference is the device string:
- PC 1: `torch_directml.device()` → `privateuseone:0`
- PC 2: `torch.device('cuda')` → `cuda:0`

For portable code, use:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
