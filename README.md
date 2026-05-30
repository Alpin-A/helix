# Helix

> 🔧 Currently under active development

A Vision-Language-Action (VLA) system for desktop robotics — combining custom embedded firmware, imitation learning, and language-conditioned control into a single cohesive system built from scratch.

> 🚧 Waiting for hardware delivery — assembly and data collection starting soon

## Overview

Helix is an end-to-end robotics project that spans three domains:
- **Firmware** — Real-time servo control written in C on an RP2040 microcontroller
- **ML** — ACT-based imitation learning policy, extended with language conditioning via a small VLM backbone
- **Dashboard** — Live web visualization of model attention, predicted actions, and task performance

## Hardware
- SO-100 robotic arm (6-DOF)
- Raspberry Pi Pico (RP2040)
- Feetech STS3215 serial bus servos
- USB cameras (overhead + wrist)

## Project Structure

    helix/
    ├── firmware/         # C code, RP2040 Pico SDK, servo driver
    ├── ml/               # Training, data collection, model architecture
    │   ├── exploration/  # Dataset analysis notebooks
    │   └── configs/      # Training configuration files
    ├── dashboard/        # React frontend + Python backend
    ├── hardware/         # Wiring diagrams, BOM, CAD files
    └── docs/             # Architecture notes and design decisions

## Progress
- [x] Phase 0: Dataset exploration complete
- [x] Phase 0: Train ACT policy on pusht simulation — final loss 0.370
- [x] Phase 1: Firmware foundations — Pico SDK setup, first C build successful
- [ ] Phase 2: Hardware assembly and control
- [ ] Phase 3: Data collection and imitation learning
- [ ] Phase 4: Language conditioning
- [ ] Phase 5: Visualization dashboard

## Setup

### ML Environment

    conda create -n helix python=3.12
    conda activate helix
    cd ml
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    pip install -e ".[pusht]"

## Findings

### Dataset Exploration (pusht)
- 25,650 frames across 206 demonstrations, averaging 124.5 frames each
- Observation and action spaces are 2D (x,y) coordinates in a 512×512 environment
- The `next.success` boolean flag uses a strict threshold and is unreliable — reward (mean max 0.892 per episode) is the better evaluation metric
