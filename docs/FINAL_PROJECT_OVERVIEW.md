# Final Project Overview

## Project Title

Real-Time Eye-Tracking-Based Deception Risk Detection Using Neural Networks

## Purpose

This project builds a controlled experimental pipeline for estimating deception-risk patterns from eye-tracking signals. It supports data collection, preprocessing, subject-independent evaluation, model comparison, replay simulation, live WebSocket inference, and run tracking.

## What the System Does

- Collects anonymous eye-tracking experiment data.
- Converts raw gaze samples into window features and sequence tensors.
- Trains and evaluates baseline and neural sequence models.
- Selects a live prototype model.
- Serves real-time deception-risk scores through a local FastAPI/WebSocket service.
- Provides a minimal dashboard and replay/client tools for testing.

## What the System Does Not Do

This system is not a universal lie detector. It outputs risk scores under a controlled experimental protocol, not final truth/lie judgments. It does not currently include real eye-tracker SDK integration, production security, database storage, or deployment packaging.

## Current Implementation Status

Steps 1-21 are implemented, including data collection, preprocessing, model training, diagnostics, live inference, dashboard, gaze adapters, dataset tracking, and a one-command pipeline runner.

## Main Pipeline

```text
raw data -> preprocessing -> windows -> subject-independent split
-> baseline models -> sequence datasets -> LSTM/GRU/TCN
-> model comparison -> model selection -> live inference -> tracking
```

## Selected Model

- Primary model: `random_forest`
- Fallback model: `gru`
- Disabled model: `lstm`
- Experimental model: `causal_tcn`

## Current Limitations

The dataset is still small, so metrics are prototype-level. Thresholds are preliminary. LSTM showed single-class prediction behavior, and Causal TCN currently has a narrow probability range.

## How to Use the Project

Start with the root `README.md`, then follow the detailed guides in `docs/`. The main rerun command is:

```bash
python -m src.pipeline.run_pipeline --mode full
```
