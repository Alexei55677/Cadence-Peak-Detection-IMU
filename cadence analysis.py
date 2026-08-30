import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray  
from scipy import signal
import pywt

SOURCE_FS_HZ = 99.84
TARGET_FS_HZ = 40.0

LOWPASS_CUTOFF_HZ = 3.2
FIR_TAPS = 120

CWT_SCALE = 10
CWT_WAVELET = "gaus2"

SAVGOL_WINDOW_LENGTH = 3
SAVGOL_POLYORDER = 0

# Starting threshold based on the method being reproduced.
# CWT coefficient scaling can differ between implementations.
PEAK_HEIGHT = 0.1

# Simplified rule for separating distinct walking bouts.
MAX_STEP_INTERVAL_S = 2.0

DATA_PATH = Path(
    r"C:\Dev\running imu project\Raw Data\device-monitor-260824-143747.tsv"
)

def load_data(path: Path) -> pd.DataFrame:
    sample_rate = 99.84
    df = pd.read_csv(path, sep = "\t")
    df["time_s"] = df["sample"] / sample_rate
    return df

def compute_acceleration_magnitude(
    df: pd.DataFrame
) -> pd.DataFrame: 
    df = df.copy()
    df["acc_norm"] = np.sqrt(
        df['ax_g'] ** 2 
        + df['ay_g'] ** 2 
        + df['az_g'] ** 2
    )
    return df 

def resample_to_40hz(
    df: pd.DataFrame
) -> pd.DataFrame:
    sample_interval = 1.0 / TARGET_FS_HZ

    new_time = np.arange(
        df["time_s"].iloc[0],
        df["time_s"].iloc[-1],
        sample_interval
    )
    new_acc_norm = np.interp(
        new_time,
        df["time_s"].to_numpy(dtype=float),
        df["acc_norm"].to_numpy(dtype=float)
    )
    return pd.DataFrame(
        {
            "time_s": new_time,
            "acc_norm": new_acc_norm,
        }
    )

def detrend_signal(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df["acc_detrended"] = signal.detrend(
        df["acc_norm"].to_numpy(dtype=float),
        type="linear"
    )
    return df

def apply_lowpass_filter(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply the 120-tap, 3.2 Hz FIR low-pass filter.
    """
    df = df.copy()
    coefficients = signal.firwin(
        numtaps=FIR_TAPS,
        cutoff=LOWPASS_CUTOFF_HZ,
        fs=TARGET_FS_HZ
    )
    filtered = signal.filtfilt(
        coefficients,
        [1.0],
        df["acc_detrended"].to_numpy(dtype=float)
    )
    df["acc_filtered"] = filtered

    return df

def apply_cwt(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply a continuous wavelet transform using a
    second-derivative Gaussian wavelet at scale 10.
    """
    df = df.copy()

    coefficients, _ = pywt.cwt(
        df["acc_filtered"].to_numpy(dtype=float),
        scales=[CWT_SCALE],
        wavelet=CWT_WAVELET,
        sampling_period=1.0 / TARGET_FS_HZ
    )
    df["acc_cwt"] = coefficients[0]

    return df

def smooth_cwt(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Smooth the CWT coefficients using a Savitzky-Golay filter.
    """
    df = df.copy()

    smoothed = signal.savgol_filter(
        df["acc_cwt"].to_numpy(dtype=float),
        window_length=SAVGOL_WINDOW_LENGTH,
        polyorder=SAVGOL_POLYORDER
    )

    df["acc_cwt_smoothed"] = smoothed #type: ignore

    return df

def detect_candidate_peaks(
    df: pd.DataFrame
) -> np.ndarray:
    """
    Detect candidate step events from the smoothed CWT signal.
    """
    peaks, properties = signal.find_peaks(
        df["acc_cwt_smoothed"].to_numpy(dtype=float),
        height=PEAK_HEIGHT
    )
    return peaks

def split_walking_periods(
        peak_times: NDArray[np.float64],
        max_step_interval: float = 2.0
) -> list[np.ndarray]:
    
    intervals = np.diff(peak_times)
    split_points = np.where(intervals > max_step_interval)[0] + 1
    walking_periods = np.split(peak_times, split_points)
    return walking_periods

def calculate_cadence(
        walking_periods: list[np.ndarray]
) -> list[float]:
    
    cadences = []
    for period in walking_periods:
        intervals = np.diff(period)
        if len(intervals) == 0:
            continue
        mean_interval = np.mean(intervals)
        cadence = 60 / mean_interval
        cadences.append(cadence)
    return cadences

# FIX: Adjusted the typing signature for strict lints
def plot_pipeline(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    peaks: np.ndarray
) -> mpl.figure.Figure: # type: ignore
    """
    Plot each stage of the signal-processing pipeline.
    """
    fig, axes = plt.subplots(
        7,
        1,
        sharex=True,
        figsize=(12, 16),
        constrained_layout=True
    )

    fig.suptitle(
        "Acceleration Magnitude Processing Pipeline",
        fontsize=16
    )

    # Original acceleration magnitude
    axes[0].plot(
        original_df["time_s"],
        original_df["acc_norm"],
        label="Original"
    )
    axes[0].set_ylabel("Magnitude (g)")
    axes[0].legend()

    # 40 Hz resampled signal
    axes[1].plot(
        processed_df["time_s"],
        processed_df["acc_norm"],
        label="Resampled to 40 Hz"
    )
    axes[1].set_ylabel("Magnitude (g)")
    axes[1].legend()

    # Detrended
    axes[2].plot(
        processed_df["time_s"],
        processed_df["acc_detrended"],
        label="Detrended"
    )
    axes[2].set_ylabel("Detrended (g)")
    axes[2].legend()

    # FIR filtered
    axes[3].plot(
        processed_df["time_s"],
        processed_df["acc_filtered"],
        label="3.2 Hz FIR low-pass"
    )
    axes[3].set_ylabel("Filtered (g)")
    axes[3].legend()

    # CWT
    axes[4].plot(
        processed_df["time_s"],
        processed_df["acc_cwt"],
        label=f"CWT ({CWT_WAVELET}, scale {CWT_SCALE})"
    )
    axes[4].set_ylabel("CWT coefficient")
    axes[4].legend()

    # Savitzky-Golay smoothing
    axes[5].plot(
        processed_df["time_s"],
        processed_df["acc_cwt_smoothed"],
        label="CWT + Savitzky-Golay"
    )
    axes[5].set_ylabel("CWT coefficient")
    axes[5].legend()

    # Detected peaks
    axes[6].plot(
        processed_df["time_s"],
        processed_df["acc_cwt_smoothed"],
        label="Smoothed CWT"
    )
    axes[6].scatter(
        processed_df["time_s"].iloc[peaks],
        processed_df["acc_cwt_smoothed"].iloc[peaks],
        color="red",
        label="Detected peaks"
    )
    axes[6].set_ylabel("CWT coefficient")
    axes[6].set_xlabel("Time (s)")
    axes[6].legend()

    return fig

def main() -> None:
    # Load and preprocess
    df = load_data(DATA_PATH)
    df = compute_acceleration_magnitude(df)
    df_40hz = resample_to_40hz(df)
    df_40hz = detrend_signal(df_40hz)
    df_40hz = apply_lowpass_filter(df_40hz)
    df_40hz = apply_cwt(df_40hz)
    df_40hz = smooth_cwt(df_40hz)

    # Detect events
    peaks = detect_candidate_peaks(df_40hz)

    peak_times = (
        df_40hz["time_s"]
        .iloc[peaks]
        .to_numpy(dtype=float)
    )

    walking_bouts = split_walking_periods(
        peak_times, max_step_interval=MAX_STEP_INTERVAL_S
    )

    cadences = calculate_cadence(walking_bouts)

    for i, cadence in enumerate(cadences, start=1):
        print(f"Walking bout {i}: {cadence:.2f} steps/min")

    # Plot results
    figure = plot_pipeline(
        df,
        df_40hz,
        peaks
    )
    plt.show()

if __name__ == "__main__":
    main()