# Roblox Up-and-Coming Chart Forecast Model

This competition entry implements a robust Python forecasting engine for the Roblox "Up and Coming" chart. The model predicts the probability distribution of game placements across the top 10 chart positions and determines the expected Top 10 ranking list over a forecast horizon.

## Model Overview

The model processes historical chart snapshots (history.ndjson) and task configuration (task.json) using several core quantitative components:

1. **Recency-Weighted Statistics**: Observations are weighted using an exponential recency half-life (HL = 10.0 hours) relative to the forecast anchor time. Games that have fallen off the chart receive appropriate rank penalties and presence dampening.
2. **Drift & Player Growth**: A weighted least-squares regression estimates rank drift over recent observations (up to 96 hours), further adjusted by concurrent player count growth (log-linear player growth slope) with player-growth tilt (KP = 25.0).
3. **Hazard & Expiry**: Survival hazard (LAM = 0.012 base rate) accounts for presence and negative player trends. Additionally, games older than 28 days from their creation timestamp are filtered out via an absolute age expiry.
4. **Horizon Diffusion & Normal CDF**: Over 12 sample points across the forecast window, diffusion grows with square root of time (S0 = 2.0, S1 = 0.7). The standard normal cumulative distribution function integrates probability mass across integer chart positions 1 through 10.
5. **Unnamed Share Allocation**: A 30% share (UNNAMED = 0.30) of each chart place probability is reserved for unnamed or new entrant games, scaling named game probabilities accordingly.

## Usage

Set the input and output directories via environment variables and run main.py:

export MAKEFUN_INPUT_DIR="/path/to/input"
export MAKEFUN_OUTPUT_DIR="/path/to/output"
python3 main.py

## Validation

When tested against the practice input dataset, the program successfully generates:
- Exactly 259 probability rows in forecast.json
- A Top 10 list matching the fingerprint: 10769671316, 10771743491, 10770575733, 10773425091, 10771928551, 10772217708, 10769941703, 10771421858, 10771881861, 10772879271
- A place-1 maximum probability of 0.0904
