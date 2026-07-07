# Knee Surgical Planning Twin

This repository contains the codebase for the Knee Surgical Planning Twin Proof of Concept (POC).

## Directory Structure

*   `src/`: Contains the main source code.
    *   `preprocessing/`: Data validation, N4 Bias Field Correction, and resampling.
*   `data/`: Directory for storing datasets (e.g., OAI-ZIB, SKI10).
*   `scripts/`: Contains scripts for batch processing and evaluation.
*   `Knee_Twin_Implementation_Plan.md`: The 9-week POC implementation plan with research-backed upgrades.

## Setup

1.  Create a virtual environment: `python -m venv venv`
2.  Activate the virtual environment:
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
