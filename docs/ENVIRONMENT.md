# Second-paper Server Environment

Official runtime selected on 2026-09-26:

- Conda environment: gume
- Python: 3.9.25
- PyTorch: 2.7.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- GPU: NVIDIA GeForce RTX 5090
- NumPy: 2.0.2
- SciPy: 1.13.0

This is the active Second-paper server environment. Upstream requirement pins are historical MSCA environment metadata and are not a reason to downgrade the RTX 5090 runtime.

Compatibility changes must be semantics-preserving only: no scientific formula change, no model architecture change, and no metric definition change.
