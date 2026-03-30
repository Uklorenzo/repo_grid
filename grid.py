# Grid Frequency Data Cleaning and Analysis
# This script loads raw grid frequency data for a specified year and month, performs cleaning to remove outliers and artifacts, and generates various plots to analyze the frequency behavior. The cleaned data is saved in a compressed format for future use.
# The script is designed to handle common file naming conventions and structures, but may require adjustments for specific datasets. It includes detailed comments and explanations for each step of the process.
# Note: Adjust the 'location', 'year', and 'month' variables at the top of the script to point to your data files. The script will attempt to find the correct file based on common naming patterns.
# Data Source: European grid frequency data, typically recorded at 1-second intervals. The script assumes the frequency is provided in mHz relative to 50 Hz (e.g., 20 mHz means 50.020 Hz). If the data appears to be in absolute Hz, it will be converted accordingly.
# The script performs the following steps:
# 1. Load the raw data from a CSV file, handling different possible schemas.
# 2. Normalize the time series to have a consistent index with 1-second intervals, filling missing timestamps with NaN.
# 3. Identify and remove outliers based on absolute frequency thresholds and large jumps.
# 4. Generate a quality plot to visualize the raw frequency data and highlight identified issues.
# 5. Plot the cleaned frequency time series and histogram.

# source of data: https://power-grid-frequency.org/database/

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from scipy import signal


# # European Data Cleaner
# ## Load Data Sets

# Location of the file: assuming it is named as year_month_Frequenz.csv
location = r'C:\Users\user\Documents\grid sim\\'
# Year
year = r'2019'
# Month
month = r'02'
# Subplot window length in days for raw Hz time-series panels
days_per_subplot = 5
year_int = int(year)
month_int = int(month)
# File name
file_name = year + '_'+ month +  '/'+ year + month + '_Frequenz.csv'
input_file = os.path.join(location, file_name)

# Try common file layouts first, then fall back to a recursive search.
candidate_files = [
    input_file,
    os.path.join(location, f'germany_{year}_{month}.csv'),
    os.path.join(location, f'germany_{year}_{month}main.csv'),
    os.path.join(location, f'{year}_{month}.csv'),
    os.path.join(location, f'{year}{month}_Frequenz.csv')
]

search_patterns = [
    os.path.join(location, '**', f'*{year}_{month}*.csv'),
    os.path.join(location, '**', f'*{year}{month}*.csv')
]

for pattern in search_patterns:
    candidate_files.extend(glob.glob(pattern, recursive=True))

# Preserve order while removing duplicates.
seen = set()
unique_candidates = []
for path in candidate_files:
    norm = os.path.normpath(path)
    if norm not in seen:
        seen.add(norm)
        unique_candidates.append(norm)

input_file = None
for path in unique_candidates:
    if os.path.isfile(path):
        input_file = path
        break

# location to save file and plot
save_to = r'C:\Users\user\Documents\grid sim\output\\'
os.makedirs(save_to, exist_ok=True)
plot_path = os.path.join(save_to, f'{year}_{month}.png')
line_hist_path = os.path.join(save_to, f'{year}_{month}_line_hist.png')
subplots_path = os.path.join(save_to, f'{year}_{month}_raw_hz_{days_per_subplot}day_subplots.png')
data_path = os.path.join(save_to, f'{year}_{month}.csv.zip')
npz_path = os.path.join(save_to, f'Data_from_cleaning_{year}_{month}.npz')
stats_subplots_path = os.path.join(save_to, f'{year}_{month}_raw_hz_{days_per_subplot}day_stats.png')
outliers_path = os.path.join(save_to, f'{year}_{month}_outliers.png')
trends_path = os.path.join(save_to, f'{year}_{month}_trends.png')
spectral_path = os.path.join(save_to, f'{year}_{month}_spectral_analysis_day1.png')
heatmap_path = os.path.join(save_to, f'{year}_{month}_heatmap_day1.png')
lag_plot_path = os.path.join(save_to, f'{year}_{month}_lag_previous_day.png')

# Plot style colors
color_series = '#1f77b4'
color_hist = '#4c78a8'
color_max = '#e45756'
color_min = '#54a24b'
color_mean = '#f58518'
color_grid = '#b0b0b0'

# Date ranges
# Build month boundaries dynamically (handles leap years automatically).
start_ts = pd.Timestamp(year=year_int, month=month_int, day=1, hour=0, minute=0, second=0)
end_ts = (start_ts + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)

# %%
#read csv files from the source path
if input_file is None:
    raise FileNotFoundError(
        "Input file not found for configured year/month. "
        f"Searched under: {os.path.normpath(location)} with year={year}, month={month}. "
        "Update 'location', 'year', and 'month' to match your data layout."
    )

print(f"Using input file: {input_file}")
df_raw = pd.read_csv(input_file, sep=',', low_memory=False)


# Normalize input schema to ['Time', 'Frequency'].
if 'Frequency' in df_raw.columns:
    # Newer files: first column is time (often Unnamed: 0), plus Frequency.
    time_col = 'Time' if 'Time' in df_raw.columns else df_raw.columns[0]
    df = pd.DataFrame({
        'Time': pd.to_datetime(df_raw[time_col], errors='coerce'),
        'Frequency': pd.to_numeric(df_raw['Frequency'], errors='coerce')
    })
else:
    # Legacy files: date/time split across columns with frequency in column 3.
    df_raw = pd.read_csv(input_file, sep=',', header=None, low_memory=False)
    if df_raw.shape[1] < 4:
        raise ValueError(
            f"Unsupported CSV schema with {df_raw.shape[1]} columns in {input_file}."
        )
    dt_str = df_raw[0].astype(str).str.strip() + ' ' + df_raw[1].astype(str).str.strip()
    df = pd.DataFrame({
        'Time': pd.to_datetime(dt_str, errors='coerce'),
        'Frequency': pd.to_numeric(df_raw[3], errors='coerce')
    })

df = df.dropna(subset=['Time'])


# Convert units only when values look like absolute Hz (~50 Hz).
freq_med = df['Frequency'].dropna().median()
if pd.notna(freq_med) and 45 <= freq_med <= 55:
    df['Frequency'] = (df['Frequency'] - 50.0) * 1000  # 60.0 for US/Japan data

# use pandas to clean the timeseries.
## First, drop all duplicates entries
df = df.drop_duplicates(subset='Time')
## Now ensure the first entry is the first second of the month and the last
## the last second of the month.

idx = pd.date_range(start_ts, end_ts, freq='s')
df = df.set_index('Time').rename_axis(None)
df = df.reindex(idx, fill_value=np.nan)

# %% Plot a 'quality plot' with the jumps, fluctuations and dead zones
fig, ax = plt.subplots(1,1, figsize=(12,3))
ax.plot(df['Frequency'], color='black')
l1 = ax.plot(df.index[df['Frequency'] > 1000], df.loc[df['Frequency'] > 1000, 'Frequency'], 'o', color='darkblue', label=r'|F|>1000')
ax.plot(df.index[df['Frequency'] < -1000], df.loc[df['Frequency'] < -1000, 'Frequency'], 'o', color='darkblue')
l2 = ax.plot(df.index[df['Frequency'].diff() > 30], df.loc[df['Frequency'].diff() > 30, 'Frequency'], 'o', color='darkorange', label=r"|F'|>30")
ax.plot(df.index[df['Frequency'].diff() < -30], df.loc[df['Frequency'].diff() < -30, 'Frequency'], 'o', color='darkorange')

cond = df['Frequency'].diff().eq(0)
grouper = (cond != cond.shift(1)).cumsum()
fill = cond & (cond.groupby(grouper).transform('size') >= 20)
l3 = ax.fill_between(df.index, -500, 500, where=fill.to_numpy(), color='purple', alpha=0.3, label=r'Plateaus F>20s')

fig.text(0.09,0.8, r'Decimals = 0', fontsize=16)
ax.set_ylim([-550,650])
ax.set_yticks([-400,-200,0,200,400])
ax.set_xlabel('Time', fontsize = 18); ax.set_ylabel('F [mHz]', fontsize = 18)
ax.legend(handles=[l1[0], l2[0], l3], loc=4, ncol=3, fontsize=14)
fig.subplots_adjust(left=0.07, bottom=0.18, right=.99, top=0.99)
fig.savefig(plot_path, dpi = 400, transparent=True)
np.savez_compressed(npz_path, cond=fill.to_numpy())

# %% Now actually clean the encountered bogus recording. Set all errors to NaN
df.loc[df['Frequency'] > 1000, 'Frequency'] = np.nan
df.loc[df['Frequency'] < -1000, 'Frequency'] = np.nan
df.loc[df['Frequency'].diff() > 20, 'Frequency'] = np.nan
df.loc[df['Frequency'].diff() < -20, 'Frequency'] = np.nan

# Find plateaus of length > 10 and replace by NaN
cond = df['Frequency'].diff().eq(0)
grouper = (cond != cond.shift(1)).cumsum()
fill = cond & (cond.groupby(grouper).transform('size') >= 20)
df.loc[fill, 'Frequency'] = np.nan

# Plot cleaned frequency as a line and histogram.
freq_clean = df['Frequency'].dropna()
freq_hz = 50.0 + (df['Frequency'] / 1000.0)
freq_clean_hz = 50.0 + (freq_clean / 1000.0)
fig2, (ax_line, ax_hist) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
ax_line.plot(df.index, freq_hz, color=color_series, linewidth=0.8)
ax_line.set_title('Frequency Time Series (Raw Hz)')
ax_line.set_xlabel('Time')
ax_line.set_ylabel('F [Hz]')
ax_line.grid(alpha=0.35, color=color_grid)

ax_hist.hist(freq_clean_hz, bins=120, color=color_hist, edgecolor='white', linewidth=0.2)
ax_hist.set_title('Frequency Histogram (Raw Hz)')
ax_hist.set_xlabel('F [Hz]')
ax_hist.set_ylabel('Count')
ax_hist.grid(alpha=0.35, color=color_grid)
fig2.savefig(line_hist_path, dpi=300)
plt.close(fig2)

# Plot raw Hz in consecutive configurable-day subplots.
window = pd.Timedelta(days=days_per_subplot)
start_all = df.index.min()
end_all = df.index.max()
n_windows = int(np.ceil((end_all - start_all + pd.Timedelta(seconds=1)) / window))

fig3, axes = plt.subplots(n_windows, 1, figsize=(13, 2.2 * n_windows), sharey=True, constrained_layout=True)
if n_windows == 1:
    axes = [axes]

for i in range(n_windows):
    window_start = start_all + i * window
    window_end = min(window_start + window - pd.Timedelta(seconds=1), end_all)
    mask = (df.index >= window_start) & (df.index <= window_end)
    axes[i].plot(df.index[mask], freq_hz[mask], color=color_series, linewidth=0.7)
    axes[i].set_ylabel('F [Hz]')
    axes[i].set_title(f'Raw Hz ({days_per_subplot}-day window): {window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}')
    axes[i].grid(alpha=0.35, color=color_grid)

axes[-1].set_xlabel('Time')
fig3.savefig(subplots_path, dpi=300)
plt.close(fig3)

# Plot 5-day window statistics (max, min, mean) in separate subplots.
window_starts = []
window_max = []
window_min = []
window_mean = []

for i in range(n_windows):
    window_start = start_all + i * window
    window_end = min(window_start + window - pd.Timedelta(seconds=1), end_all)
    mask = (df.index >= window_start) & (df.index <= window_end)
    segment = freq_hz[mask].dropna()
    if segment.empty:
        continue
    window_starts.append(window_start)
    window_max.append(segment.max())
    window_min.append(segment.min())
    window_mean.append(segment.mean())

fig4, (ax_max, ax_min, ax_mean) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
ax_max.plot(window_starts, window_max, marker='o', color=color_max, linewidth=1.6)
ax_max.set_title(f'Max Frequency per {days_per_subplot}-Day Window')
ax_max.set_ylabel('Max [Hz]')
ax_max.grid(alpha=0.35, color=color_grid)
for x, y in zip(window_starts, window_max):
    ax_max.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', color=color_max, fontsize=8)

ax_min.plot(window_starts, window_min, marker='o', color=color_min, linewidth=1.6)
ax_min.set_title(f'Min Frequency per {days_per_subplot}-Day Window')
ax_min.set_ylabel('Min [Hz]')
ax_min.grid(alpha=0.35, color=color_grid)
for x, y in zip(window_starts, window_min):
    ax_min.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', color=color_min, fontsize=8)

ax_mean.plot(window_starts, window_mean, marker='o', color=color_mean, linewidth=1.6)
ax_mean.set_title(f'Mean Frequency per {days_per_subplot}-Day Window')
ax_mean.set_ylabel('Mean [Hz]')
ax_mean.set_xlabel('Window Start Time')
ax_mean.grid(alpha=0.35, color=color_grid)
for x, y in zip(window_starts, window_mean):
    ax_mean.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                     ha='center', va='bottom', color=color_mean, fontsize=8)

fig4.savefig(stats_subplots_path, dpi=300)
plt.close(fig4)

# Outlier detection: use z-score to flag values beyond 3 standard deviations.
freq_clean_series = df['Frequency'].dropna()
z_scores = np.abs((freq_clean_series - freq_clean_series.mean()) / freq_clean_series.std())
outlier_mask = z_scores > 3
outlier_indices = freq_clean_series[outlier_mask].index
freq_hz_clean = 50.0 + (freq_clean_series / 1000.0)

fig5, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
ax.plot(freq_clean_series.index, freq_hz_clean, color=color_series, linewidth=0.7, label='Cleaned Frequency', alpha=0.8)
if len(outlier_indices) > 0:
    ax.scatter(outlier_indices, 50.0 + (df.loc[outlier_indices, 'Frequency'] / 1000.0), color=color_max, s=60, label='Outliers (|z| > 3)', zorder=5)
ax.set_title('Outlier Detection (Z-Score > 3σ)')
ax.set_xlabel('Time')
ax.set_ylabel('F [Hz]')
ax.legend(loc='upper right')
ax.grid(alpha=0.35, color=color_grid)
fig5.savefig(outliers_path, dpi=300)
plt.close(fig5)
print(f'Found {len(outlier_indices)} outliers.')

# Trend analysis: plot raw frequency with rolling mean overlay.
rolling_mean_1h = df['Frequency'].rolling(window=3600, center=True, min_periods=1).mean()
rolling_mean_1h_hz = 50.0 + (rolling_mean_1h / 1000.0)

fig6, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
ax.plot(df.index, freq_hz, color=color_series, linewidth=0.4, label='Raw Frequency', alpha=0.6)
ax.plot(df.index, rolling_mean_1h_hz, color=color_mean, linewidth=1.8, label='1-Hour Rolling Mean', zorder=3)
ax.set_title('Frequency Trend (1-Hour Rolling Mean)')
ax.set_xlabel('Time')
ax.set_ylabel('F [Hz]')
ax.legend(loc='upper right')
ax.grid(alpha=0.35, color=color_grid)
fig6.savefig(trends_path, dpi=300)
plt.close(fig6)

# One-day heatmap (day 1): hour vs minute with color as frequency in Hz.
day1_start_heat = df.index.min().normalize()
day1_end_heat = day1_start_heat + pd.Timedelta(days=1)
day1_freq_hz_full = freq_hz[(df.index >= day1_start_heat) & (df.index < day1_end_heat)]
day1_minute = day1_freq_hz_full.resample('1min').mean()
minute_grid = pd.date_range(day1_start_heat, day1_end_heat - pd.Timedelta(minutes=1), freq='1min')
day1_minute = day1_minute.reindex(minute_grid)

if len(day1_minute) == 24 * 60:
    heat_matrix = day1_minute.to_numpy().reshape(24, 60)
    fig8, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax.imshow(heat_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title(f'Frequency Heatmap (Raw Hz) - {day1_start_heat:%Y-%m-%d}')
    ax.set_xlabel('Minute of Hour')
    ax.set_ylabel('Hour of Day')
    ax.set_xticks(np.arange(0, 60, 10))
    ax.set_yticks(np.arange(0, 24, 2))
    cbar = fig8.colorbar(im, ax=ax)
    cbar.set_label('F [Hz]')
    fig8.savefig(heatmap_path, dpi=300)
    plt.close(fig8)

# Lag plot: compare each day with previous day using 1-minute averages.
freq_1min = freq_hz.resample('1min').mean()
lag_df = pd.DataFrame({'FHz': freq_1min})
lag_df['date'] = lag_df.index.date
lag_df['minute_of_day'] = lag_df.index.hour * 60 + lag_df.index.minute
pivot = lag_df.pivot(index='minute_of_day', columns='date', values='FHz')
dates = list(pivot.columns)

if len(dates) >= 2:
    lag_x_all = []
    lag_y_all = []
    day_corrs = []
    day_labels = []

    for i in range(1, len(dates)):
        prev_day = pivot[dates[i - 1]]
        curr_day = pivot[dates[i]]
        valid = prev_day.notna() & curr_day.notna()
        if valid.sum() < 10:
            continue
        x = prev_day[valid].to_numpy()
        y = curr_day[valid].to_numpy()
        lag_x_all.append(x)
        lag_y_all.append(y)
        day_corrs.append(np.corrcoef(x, y)[0, 1])
        day_labels.append(f'{dates[i - 1]} -> {dates[i]}')

    if lag_x_all:
        x_all = np.concatenate(lag_x_all)
        y_all = np.concatenate(lag_y_all)
        min_xy = np.nanmin([x_all.min(), y_all.min()])
        max_xy = np.nanmax([x_all.max(), y_all.max()])

        fig9, (ax_lag, ax_corr) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
        ax_lag.scatter(x_all, y_all, s=4, alpha=0.25, color=color_series, edgecolors='none')
        ax_lag.plot([min_xy, max_xy], [min_xy, max_xy], color=color_max, linewidth=1.4, linestyle='--', label='y = x')
        ax_lag.set_title('Lag Plot: Current Day vs Previous Day (1-min averages)')
        ax_lag.set_xlabel('Previous Day F [Hz]')
        ax_lag.set_ylabel('Current Day F [Hz]')
        ax_lag.legend(loc='upper left')
        ax_lag.grid(alpha=0.35, color=color_grid)

        ax_corr.plot(range(len(day_corrs)), day_corrs, marker='o', linewidth=1.4, color=color_mean)
        ax_corr.set_title('Daily Lag Correlation (Current vs Previous Day)')
        ax_corr.set_xlabel('Day Pair Index')
        ax_corr.set_ylabel('Correlation')
        ax_corr.set_ylim(-1.0, 1.0)
        ax_corr.grid(alpha=0.35, color=color_grid)

        fig9.savefig(lag_plot_path, dpi=300)
        plt.close(fig9)

# Spectral analysis for the first day of the month.
day1_start = df.index.min()
day1_end = day1_start + pd.Timedelta(days=1)
day1_mask = (df.index >= day1_start) & (df.index < day1_end)
day1_data = df.loc[day1_mask, 'Frequency'].dropna()

if len(day1_data) > 1:
    sampling_rate = 1.0
    day1_freq_hz = 50.0 + (day1_data / 1000.0)
    freqs, pxx = signal.welch(day1_freq_hz.values, fs=sampling_rate, nperseg=min(1024, len(day1_data)))
    pxx_db_hz = 10.0 * np.log10(np.maximum(pxx, 1e-20))
    
    fig7, (ax_ts, ax_spec) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    ax_ts.plot(day1_data.index, day1_freq_hz, color=color_series, linewidth=0.8)
    ax_ts.set_title(f'Day 1 Time Series: {day1_start:%Y-%m-%d}')
    ax_ts.set_ylabel('F [Hz]')
    ax_ts.grid(alpha=0.35, color=color_grid)
    
    ax_spec.plot(freqs, pxx_db_hz, color=color_mean, linewidth=1.2)
    ax_spec.set_title('Power Spectral Density (Welch, dB/Hz)')
    ax_spec.set_xlabel('Frequency [Hz]')
    ax_spec.set_ylabel('PSD [dB/Hz]')
    ax_spec.grid(alpha=0.35, color=color_grid)
    fig7.savefig(spectral_path, dpi=300)
    plt.close(fig7)

# %% Save data into a zipped csv. location is save_to

df.to_csv(data_path, float_format='%.3f',
    compression=dict(method='zip', archive_name=year+'_'+month+'.csv'))

print('Cleaning complete.')
print('Saved plot:', plot_path)
print('Saved line+hist plot:', line_hist_path)
print(f'Saved {days_per_subplot}-day subplots:', subplots_path)
print(f'Saved {days_per_subplot}-day stats subplots:', stats_subplots_path)
print('Saved outlier plot:', outliers_path)
print('Saved trend plot:', trends_path)
print('Saved one-day heatmap:', heatmap_path)
print('Saved lag plot:', lag_plot_path)
print('Saved spectral analysis plot:', spectral_path)
print('Saved data:', data_path)
print('Saved diagnostics:', npz_path)


# %% Extras: Removing the extra hour in October due to daylight savings
## 2014
# df[1][2174400-2*3600:2174400-1*3600] = list(map(str,pd.date_range('02:00:00','02:59:59', freq = 'S').time))
# df = df.drop(df.index[2174400-1*3600:2174400-0*3600])