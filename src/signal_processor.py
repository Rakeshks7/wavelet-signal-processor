import numpy as np
import pandas as pd
import pywt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaveletDenoiser:
    
    def __init__(self, wavelet: str = 'db4', level: int = 2):
        self.wavelet = wavelet
        self.level = level

    def denoise(self, data: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        cD1 = coeffs[-1]
        sigma = median_abs_deviation(cD1, scale='normal')

        N = len(data)
        threshold = sigma * np.sqrt(2 * np.log(N))

        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]

        reconstructed_signal = pywt.waverec(coeffs, self.wavelet)

        return reconstructed_signal[:len(data)]

class DataFetcher:
    
    @staticmethod
    def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker}. Check ticker symbol or dates.")
        return df

class StrategyEngine:
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, price_col: str = 'Close', denoised_col: str = 'Denoised') -> pd.DataFrame:
        logging.info("Generating strategy signals...")

        df['Signal_Trigger'] = df[denoised_col].rolling(window=5).mean()

        df['Position'] = 0
        df.loc[df[denoised_col] > df['Signal_Trigger'], 'Position'] = 1
        df.loc[df[denoised_col] < df['Signal_Trigger'], 'Position'] = -1

        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Log_Returns']

        df['Cum_Market_Returns'] = np.exp(df['Log_Returns'].cumsum())
        df['Cum_Strategy_Returns'] = np.exp(df['Strategy_Returns'].cumsum())
        
        return df

class Visualizer:
    
    @staticmethod
    def plot_results(df: pd.DataFrame, ticker: str):
        logging.info("Rendering visualizations...")
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df.index, df['Close'], color='gray', alpha=0.5, label='Raw Price (Noise)', lw=1)
        ax1.plot(df.index, df['Denoised'], color='cyan', label='Denoised Signal (True Trend)', lw=2)
        ax1.plot(df.index, df['Signal_Trigger'], color='magenta', linestyle='--', label='Trigger Line (Fast MA of Denoised)', lw=1.5)
        
        ax1.set_title(f"Wavelet Denoising & Trend Extraction: {ticker}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # Bottom Plot: Cumulative Returns
        ax2.plot(df.index, df['Cum_Market_Returns'], color='gray', alpha=0.7, label='Buy & Hold Returns')
        ax2.plot(df.index, df['Cum_Strategy_Returns'], color='green', label='Wavelet Strategy Returns', lw=2)
        
        ax2.set_title("Strategy Performance (Log Returns)", fontsize=12)
        ax2.set_ylabel("Cumulative Multiplier")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.show()

def main():
    TICKER = "BTC-USD"
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"
    WAVELET_TYPE = "db4"  # Daubechies 4 is excellent for financial time series
    DECOMPOSITION_LEVEL = 3

    fetcher = DataFetcher()
    df = fetcher.fetch_data(TICKER, START_DATE, END_DATE)

    raw_prices = df['Close'].values.flatten()
    
    denoiser = WaveletDenoiser(wavelet=WAVELET_TYPE, level=DECOMPOSITION_LEVEL)
    denoised_prices = denoiser.denoise(raw_prices)

    df['Denoised'] = denoised_prices

    engine = StrategyEngine()
    df = engine.generate_signals(df, price_col='Close', denoised_col='Denoised')

    Visualizer.plot_results(df, TICKER)

if __name__ == "__main__":
    main()