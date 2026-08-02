# ⚛️ ZKAEDI QUANTUM CORE & OBSERVATORY

[![Python Version](https://img.shields.io/badge/python-3.10%2B-00f3ff.svg?style=for-the-badge&logo=python)](https://python.org)
[![Build Status](https://img.shields.io/badge/build-100%25%20GREEN-00ff99.svg?style=for-the-badge)](tools/verify_all_algorithms.py)
[![License](https://img.shields.io/badge/license-MIT-a855f7.svg?style=for-the-badge)](LICENSE)
[![OpenQASM 3.0](https://img.shields.io/badge/interop-OpenQASM%203.0-ffd700.svg?style=for-the-badge)](src/quantum_core/openqasm.py)
[![ZCC C99 Engine](https://img.shields.io/badge/codegen-ZCC%20C99-ff0055.svg?style=for-the-badge)](src/quantum_core/zcc_codegen.py)

> Open-source Python stack for fault-tolerant quantum compilation — QEC (surface code + qLDPC resource models), magic-state distillation scaffolding, Clifford+T synthesis with T-count budgeting, OpenQASM 3 I/O, and C99 codegen — with multi-suite tests, CLI audit, and interactive WebGL observatories.

---

## ⚡ Quickstart & What It Runs Today

### 1. Run Automated Test Suite (21 Test Suites)
```bash
python tools/verify_all_algorithms.py
```

### 2. Execute Cryptographic Audit CLI
```bash
python -m quantum_core audit
```

### 3. Run End-to-End Compilation Pipeline
Execute $H_0 \to \text{Optimization} \to \text{MPS} \to \text{Surface Code MWPM} \to \text{14-to-2 Distillation Scaffolding} \to \text{ZCC C Codegen}$:
```bash
python examples/end_to_end_compilation_pipeline.py
```

---

## 🌐 Interactive WebGL Observatories

Single-file, zero-dependency HTML/WebGL applications for compiler intuition and visual analysis:

| Visual Studio | Description | Direct Link |
|---|---|---|
| 👑 **Sovereign Observatory Hub** | Master Landing Portal linking all 4 WebGL studios under one cyberpunk HUD. | [index.html](docs/index.html) |
| 🔮 **ZKAEDI Holographic Studio v2.0** | 3D WebGL Bloch Sphere visualizer, 300-particle quantum lattice, Surface Code [[25,1,5]] MWPM decoder, and live ZCC C99 C code generator. | [quantum_core_holographic_studio.html](docs/quantum_core_holographic_studio.html) |
| 🤖 **AI Multi-Model Arena v1.0** | Interactive 4-LLM battle arena (GPT-4o, Claude 3.5 Sonnet, DeepSeek-R1, Gemma 2) with 3-Judge automated consensus panel & radar charts. | [multi_model_arena.html](docs/multi_model_arena.html) |
| 🌿 **AURA Health OMEGA v4.0** | Supreme Clinical Companion featuring a 3D Anatomical Body Map, AI Drug Safety Checker, Mood Bio-Tracker, TDEE Calculator, 432Hz Soundscape, and printable medical report exporter. | [aura_health_companion.html](docs/aura_health_companion.html) |
| 🎮 **Pokémon RL Model Health Center** | 3D WebGL Pokeball Tray Healing Machine featuring Nurse Joy AI, live RL reward engine, policy gradient restoration, 6-agent party level-ups, and synthesized Pokecenter healing chime. | [pokemon_rl_health_center.html](docs/pokemon_rl_health_center.html) |
| 🪐 **ORBITAL PRIME v2.0** | 3D N-Body Velocity Verlet Gravitational Simulator featuring Sol System, Alpha Centauri Binary, Trappist-1 Exoplanets, Supermassive Black Hole Accretion Disk, Doppler Pitch Synth, and CSV telemetry exporter. | [orbital_prime_nbody_studio.html](docs/orbital_prime_nbody_studio.html) |
| ⚡ **Cyberpunk Onboarding Suite** | Neon HUD walkthrough, live MPS visualizer, particle canvas physics, and Web Audio API synthesizer. | [quantum_core_cyberpunk_onboarding.html](docs/quantum_core_cyberpunk_onboarding.html) |

---

## 🛠️ Pipeline Architecture & Resource Models

- **Stabilizer QEC & Resource Modeling**: Rotated surface-code specs, syndrome extraction, and Bivariate Bicycle qLDPC Code [[156,12,12]] vs $2d^2$-style physical qubit resource comparison models.
- **MWPM Decoding & Distillation Scaffolding**: Edmonds' Blossom MWPM syndrome decoding and Bravyi–Haah 14-to-2 magic-state distillation protocol scaffolding for logical $T$-gate budgeting.
- **Clifford+T Synthesis Engine**: Decomposes continuous $R_z(\theta)$ rotations into Clifford+T sequences with precision-scaled $T$-count budgets ($N_T \approx 3 \log_2(1/\epsilon)$).
- **Scale-Out MPS Tensor Network Simulator**: Simulates 1D circuits via SVD singular value bond dimension truncation ($\chi=64$) keeping memory requirements in MB-scale limits.
- **Vendor Interoperability & C Codegen**: Exports OpenQASM 3.0 compatible with industry toolchains and emits pure C99 C source code for ZCC compilation.

---

## 📦 Core Architecture (25 Modules)

```text
src/quantum_core/
├── types.py          # Data contracts (GateOperation, CircuitSpec)
├── gates.py          # Immutable unitary gate library (H, X, Y, Z, S, T, CX, Rz)
├── validation.py     # Qubit range bounds checks & gate alias resolution
├── circuit.py        # Variational symbolic parameter binder (VQE / QNN)
├── simulator.py      # O(2^N) statevector simulator engine
├── measurement.py    # Born-rule probability sampling & state collapse
├── density_matrix.py # Mixed-state density operator & Kraus channels
├── mps.py            # MPS Tensor Network simulator with SVD bond truncation (chi=64)
├── stabilizers.py    # Steane [[7,1,3]], Surface [[9,1,3]] & [[25,1,5]] generators
├── decoding.py       # Minimum-Weight Perfect Matching (MWPM) syndrome decoder
├── triorthogonal.py # Binary triorthogonality checks (15-to-1 & 14-to-2)
├── magic_states.py   # Bravyi-Haah 14-to-2 magic state distillation factory
├── qldpc.py          # qLDPC Bivariate Bicycle Code [[156,12,12]] model
├── optimization.py   # Peep-hole pass (self-inverse cancellation & rotation merging)
├── routing.py        # 2D Grid Manhattan distance SWAP routing pass
├── synthesis.py      # Clifford+T synthesis engine (precision scaling 3*log2(1/eps))
├── openqasm.py       # OpenQASM 3.0 Exporter & Importer (IBM / AWS Braket ready)
├── zcc_codegen.py    # Pure C99 C code generator for ZCC compiler execution
├── error_models.py   # Depolarizing noise, bit/phase flips & thermal relaxation
├── benchmarking.py   # Randomized Benchmarking (RB) sequence generator & decay fitter
├── evidence.py       # Cryptographic SHA-256 evidence bundle generator
├── registry.py       # Discovery catalog manifest auditor & path resolver
├── cli.py            # Unified Audit CLI runner
├── __main__.py       # Package entry point (python -m quantum_core audit)
└── __init__.py       # Package exports & version declaration ("1.0.0")
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.