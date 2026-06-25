# Wavelet Signal Processor: Trend Extraction via DWT

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Production](https://img.shields.io/badge/Status-Production-success.svg)

Moving averages lag. Fourier Transforms lose time domain information. **Wavelet Signal Processor** utilizes the Discrete Wavelet Transform (DWT) to de-noise financial time-series data while preserving sharp structural breaks (like market gaps or sudden volatility spikes). 

This repository provides a production-grade pipeline to strip Gaussian noise from asset prices, extract the true underlying trend, and build zero-lag momentum crossover strategies.

## Key Features

* **Wavelet Decomposition:** Utilizes Daubechies (`db4`) wavelets to break down price structures into Approximation (Trend) and Detail (Noise) coefficients.
* **VisuShrink Thresholding:** Dynamically estimates noise variance using Median Absolute Deviation (MAD) and applies Soft Thresholding to clean the signal.
* **Zero-Lag Reconstruction:** Rebuilds the price series without the phase lag inherent in standard moving averages (SMA/EMA).
* **Strategy Engine:** Includes a foundational momentum backtester to generate Alpha from the mathematically cleaned trend line.

## Tech Stack
* **Core:** Python, `numpy`, `pandas`
* **Signal Processing:** `PyWavelets`, `scipy`
* **Data & Viz:** `yfinance`, `matplotlib`

## Disclaimer
Not Financial Advice. This software is provided for educational and research purposes only. The quantitative strategies and signal processing methodologies demonstrated here do not constitute investment advice. Trading financial markets involves significant risk. Always conduct your own rigorous backtesting and forward-testing before deploying capital.
