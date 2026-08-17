# AGENTS.md - AI Operating Guidelines & Project Protocol
# Repository: HDML (Hierarchical Decision Mamba-Liquid)
# Standard: Linux Foundation / Agentic AI Foundation Specification

This document provides definitive instructions, operational boundaries, and Python/PyTorch engineering standards for all AI coding agents working within this repository.

---

## 1. Core Operating Principles

### 1.1 Zero Hallucination & Empirical Verification
- **Empirical Validation Required**: Never declare a task completed without running empirical execution using `/data/HDML_Model/.venv/bin/python` on active hardware (`cuda:0`).
- **No Speculative Coding**: Inspect file contents, library versions, and execution logs directly. Do not guess API arguments or tensor shape contracts.
- **Active Code Reference**: Always cross-reference imports against installed libraries in `.venv` (`torch`, `mamba_ssm`, `ncps`, `gymnasium`, `minari`, `timm`).

### 1.2 Absolute Transparency & Anti-Concealment
- **NO Silent Failures**: Never write empty `except:` or `except Exception: pass` blocks that swallow errors. All exceptions must be explicitly caught, logged with full stack trace, and properly handled or re-raised.
- **NO Fake Mocks**: Never create dummy assertions or mock objects to simulate successful test passes.
- **NO Placeholder Code**: Never leave `# TODO: implement later`, `pass`, `...`, or unhandled stubs inside active modules.
- **Log Full Tracebacks**: On runtime or compilation errors, expose the entire traceback for debugging.

---

## 2. Python Coding & Clean Architecture Standards

### 2.1 Type Hints & Function Signatures
- **Strict Typing**: Enforce explicit type annotations on all function signatures, methods, and class attributes (PEP 484, PEP 585, PEP 604).
- **Forward References**: Include `from __future__ import annotations` at the top of all new Python modules.
- **Tensor Typing**: Type all tensor arguments explicitly as `torch.Tensor` or `Tensor`. Avoid generic `Any`.
- **Compound Types**: Use modern Python 3.11 union syntax (`T | None`, `list[int]`, `tuple[int, ...]`, `dict[str, torch.Tensor]`).

### 2.2 Defensive Programming & Python Pitfalls
- **NO Mutable Default Arguments**: Never use mutable objects (`[]`, `{}`) as default parameters. Use `def func(param: list[int] | None = None) -> None:` and initialize inside.
- **File System & Paths**: Prefer `pathlib.Path` over `os.path` for all filesystem manipulations.
- **Context Managers**: Always use `with open(...) as f:` for I/O operations to guarantee proper descriptor closure.
- **Docstrings & Shape Contracts**: All neural network layers and functions must document expected tensor dimensions in docstrings (e.g., `Input: (Batch, Seq_Len, Dim_In) -> Output: (Batch, Seq_Len, Dim_Out)`).

---

## 3. PyTorch & Deep Learning Engineering Standards

### 3.1 Device & Memory Management
- **Device-Agnostic Placement**: Modules must accept `device: torch.device | str = "cuda"` and dynamically assign child layers and buffers via `.to(device)`.
- **Inference Efficiency**: Always wrap evaluation, rollout simulation, and validation passes in `torch.inference_mode()` or `torch.no_grad()` to prevent VRAM memory accumulation.
- **Cache Management**: Call `torch.cuda.empty_cache()` at the end of memory-intensive evaluation loops.
- **Constructors**: Prefer `torch.tensor(..., device=device)` and `torch.zeros/ones` over legacy `torch.Tensor()` constructors.

### 3.2 Gradient & Numerical Stability
- **Shape Assertions**: Enforce explicit assertions on tensor dimensions before critical matrix multiplications, linear projections, and cross-modal concatenation:
  ```python
  assert vision_feats.shape[-1] == d_model, f"Expected dim {d_model}, got {vision_feats.shape[-1]}"
  ```
- **Gradient Flow Verification**: Always verify `.grad is not None` and test that gradients propagate through both Mamba and Liquid layers via `.backward()`.
- **NaN/Inf Safeguards**: Check `torch.isfinite(loss).all()` before calling `optimizer.step()` to prevent parameter corruption.
- **Reproducibility**: Set seeds deterministically when writing tests:
  ```python
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  ```

---

## 4. Hardware & Compilation Protocol

### 4.1 Target Environment
- **Hardware**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM, Compute Capability `8.9` / Ada Lovelace).
- **System CUDA**: CUDA Toolkit 13.2 (`/usr/local/cuda-13.2/bin/nvcc`).
- **Python Runtime**: Python 3.11.15 in `/data/HDML_Model/.venv`.

### 4.2 CUDA SSM Compilation
- Custom C++/CUDA kernel packages (`mamba-ssm`, `causal-conv1d`) **must** be compiled with `--no-build-isolation` to link directly with PyTorch CUDA headers:
  ```bash
  export CUDA_HOME="/usr/local/cuda-13.2"
  export TORCH_CUDA_ARCH_LIST="8.9"
  export MAX_JOBS=8

  pip install causal-conv1d --no-build-isolation
  pip install mamba-ssm --no-build-isolation
  ```

---

## 5. Three-Tier Operational Protocol

### 5.1 ALWAYS
- Execute and validate code on the active GPU before reporting task completion.
- Test forward output shapes and backward gradient flow on real tensors.
- Keep `README.md`, `SETUP_AND_DOCS.md`, and `research.md` strictly synchronized.

### 5.2 ASK FIRST
- Adding new external packages or changing library versions in `requirements.txt`.
- Refactoring or altering core mathematical formulations (ODE parameters, SSM discretization).
- Renaming or deleting core architectural files.

### 5.3 NEVER
- Never write empty `except:` blocks that suppress errors.
- Never mock test results or bypass real GPU execution.
- Never leave `# TODO`, `pass`, or incomplete code blocks.
