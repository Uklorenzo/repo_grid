import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. FILE PATHS
# =========================
file_path = r"C:\Users\user\Documents\grid sim\germany_2011_12.csv"
save_to = r"C:\Users\user\grid sim\output\\"
os.makedirs(save_to, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(file_path, sep=None, engine='python')

# Clean column names
df.columns = df.columns.str.strip()

print("Columns found:", df.columns)

# If only one column → fix separator
if len(df.columns) == 1:
    df = pd.read_csv(file_path, sep='\t')
    df.columns = df.columns.str.strip()

# Rename dynamically
df = df.rename(columns={
    df.columns[0]: 'Time',
    df.columns[1]: 'Frequency'
})

# Convert types
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')

# =========================
# 3. DETECT SAMPLING RATE
# =========================
print("Detecting sampling rate...")

time_counts = df['Time'].value_counts().sort_index()

# If many duplicate timestamps → high-frequency data collapsed
duplicates_per_timestamp = int(time_counts.median())

print(f"Approx samples per timestamp: {duplicates_per_timestamp}")

if duplicates_per_timestamp > 1:
    inferred_freq = f"{int(1/duplicates_per_timestamp*1e6)}U"  # microseconds
    print("High-frequency data detected (collapsed timestamps)")
else:
    inferred_freq = "S"  # default to seconds

print(f"Using frequency: {inferred_freq}")

# =========================
# 4. REBUILD TIME INDEX
# =========================
df = df.sort_values('Time').reset_index(drop=True)

start_time = df['Time'].iloc[0]

# If timestamps are identical → rebuild evenly spaced time
if df['Time'].nunique() == 1:
    print("Rebuilding timeline from single timestamp...")
    df['Time'] = pd.date_range(start=start_time, periods=len(df), freq='S')

else:
    # If timestamps exist but duplicates present → spread them
    new_times = []
    for t, group in df.groupby('Time'):
        n = len(group)
        if n == 1:
            new_times.append(t)
        else:
            spread = pd.date_range(start=t, periods=n, freq='S')
            new_times.extend(spread)

    df['Time'] = new_times

# Remove duplicates safely
df = df.drop_duplicates(subset='Time')

# =========================
# 5. NORMALIZE FREQUENCY
# =========================
df['Frequency'] = (df['Frequency'] - 50.) * 1000

# =========================
# 6. BUILD FULL TIME RANGE
# =========================
year = df['Time'].dt.year.iloc[0]
month = df['Time'].dt.month.iloc[0]

start = pd.Timestamp(year, month, 1, 0, 0, 0)
end = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(hours=23, minutes=59, seconds=59)

full_index = pd.date_range(start, end, freq='S')

df = df.set_index('Time').reindex(full_index)

# =========================
# 7. CLEANING LOGIC
# =========================
# Extreme values
df.loc[df['Frequency'] > 1000, 'Frequency'] = np.nan
df.loc[df['Frequency'] < -1000, 'Frequency'] = np.nan

# Sudden jumps
df.loc[df['Frequency'].diff() > 20, 'Frequency'] = np.nan
df.loc[df['Frequency'].diff() < -20, 'Frequency'] = np.nan

# Plateaus detection
cond = df['Frequency'].diff() == 0
groups = (cond != cond.shift()).cumsum()
sizes = df.groupby(groups)['Frequency'].transform('size')

df.loc[(cond) & (sizes >= 20), 'Frequency'] = np.nan

# =========================
# 8. QUALITY PLOT
# =========================
fig, ax = plt.subplots(figsize=(12,3))

ax.plot(df.index, df['Frequency'], color='black', label='Frequency')

# Highlight anomalies
ax.plot(df[df['Frequency'] > 1000], 'o', label='>1000')
ax.plot(df[df['Frequency'] < -1000], 'o')
ax.plot(df[df['Frequency'].diff() > 20], 'o', label='Jump >20')
ax.plot(df[df['Frequency'].diff() < -20], 'o')

ax.set_ylim([-550, 650])
ax.set_xlabel('Time')
ax.set_ylabel('F [mHz]')
ax.legend()

plt.savefig(save_to + 'quality_plot.png', dpi=300)

# =========================
# 9. SAVE OUTPUT
# =========================
df.to_csv(save_to + 'cleaned_data.csv')

print("✅ Done. Cleaned data + plot saved.")