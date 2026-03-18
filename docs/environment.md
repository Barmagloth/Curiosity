# Environment Setup

## Hardware

- **GPU**: AMD Radeon 780M (RDNA 3 iGPU, gfx1103) -- integrated in Ryzen 7000/8000 series
- **OS**: Windows 11 Pro 10.0.26100

## Software

- **Python**: 3.13.2 (system install)
- **Virtual environment**: `.venv` in project root (`python -m venv .venv`)
- **PyTorch**: 2.10.0+cpu
- **Backend**: CPU-only

### Why CPU-only for .venv?

The AMD Radeon 780M is an integrated GPU. The available acceleration paths on Windows are:

1. **ROCm** -- not available on Windows for iGPUs (ROCm Windows support is limited to discrete GPUs like RX 7900 series, and even then it is experimental).
2. **DirectML** (`torch-directml`) -- the correct backend for AMD iGPUs on Windows, but as of March 2026 it does not support Python 3.13. The latest release requires Python <=3.12.
3. **CPU** -- fallback, works on all Python versions.

The main `.venv` stays on Python 3.13 + CPU for maximum compatibility. GPU work uses `.venv-gpu` (see below).

## GPU virtual environment (.venv-gpu)

- **Python**: 3.12.10 (installed via `winget install Python.Python.3.12`)
- **Virtual environment**: `.venv-gpu` in project root (`py -3.12 -m venv .venv-gpu`)
- **PyTorch**: 2.4.1+cpu (pinned by torch-directml)
- **Backend**: DirectML via `torch-directml 0.2.5.dev240914`
- **GPU detected**: AMD Radeon(TM) 780M

### Replication instructions (GPU venv)

```bash
cd C:\Users\alexeia\Lab_A_Internal\Curiosity
py -3.12 -m venv .venv-gpu
.venv-gpu\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torch-directml
pip install numpy scipy matplotlib
```

Note: `torch-directml` will downgrade torch to 2.4.1 (its pinned dependency). This is expected.

### Verification (GPU venv)

```bash
.venv-gpu\Scripts\python -c "import torch; import torch_directml; dml = torch_directml.device(); t = torch.rand(3,3, device=dml); print(t); print('Device:', torch_directml.device_name(0))"
# Expected: tensor on privateuseone:0, Device: AMD Radeon(TM) 780M
```

### DirectML benchmark results (2026-03-18)

Matrix multiply, CPU vs DirectML on AMD Radeon 780M:

| Size | CPU | DirectML | Speedup |
|------|-----|----------|---------|
| 512x512 | 1.0 ms | 1.4 ms | 0.74x |
| 1024x1024 | 5.6 ms | 4.9 ms | 1.14x |
| 2048x2048 | 43.1 ms | 23.5 ms | 1.83x |
| 4096x4096 | 335.0 ms | 126.7 ms | 2.64x |

DirectML shows clear speedup for matrices >= 1024. Small matrices are dominated by CPU-GPU transfer overhead.

### Installed packages -- GPU venv (pip freeze)

```
contourpy==1.3.3
cycler==0.12.1
filelock==3.25.2
fonttools==4.62.1
fsspec==2026.2.0
Jinja2==3.1.6
kiwisolver==1.5.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.3
packaging==26.0
pillow==12.1.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy==1.17.1
setuptools==82.0.1
six==1.17.0
sympy==1.14.0
torch==2.4.1
torch-directml==0.2.5.dev240914
torchvision==0.19.1
typing_extensions==4.15.0
```

## CPU virtual environment (.venv)

### Installed packages -- CPU venv (pip freeze)

```
contourpy==1.3.3
cycler==0.12.1
filelock==3.25.2
fonttools==4.62.1
fsspec==2026.2.0
Jinja2==3.1.6
kiwisolver==1.5.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.3
packaging==26.0
pillow==12.1.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy==1.17.1
setuptools==82.0.1
six==1.17.0
sympy==1.14.0
torch==2.10.0
torchvision==0.25.0
typing_extensions==4.15.0
```

## Replication instructions

```bash
cd C:\Users\alexeia\Lab_A_Internal\Curiosity
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision numpy scipy matplotlib
```

## Verification

```bash
.venv\Scripts\python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Expected: 2.10.0+cpu, CUDA: False

.venv\Scripts\python -c "import numpy, scipy, matplotlib; print('OK')"
# Expected: OK
```

## Experiment import test

Both key experiment files import successfully with the venv:

- `experiments/exp07_gate/exp07b_twostage.py` -- requires `numpy`, `scipy.ndimage`, `json`, plus internal cross-imports from exp04/exp05/exp06
- `experiments/exp08_schedule/exp08v5_schedule.py` -- requires `numpy`, `scipy`, `matplotlib` (self-contained)
