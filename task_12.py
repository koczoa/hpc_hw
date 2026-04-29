import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results_all.csv"   # adjust if needed
HIST_OUT = "hist_mean_temp.png"

# Load
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} buildings from {CSV_PATH}")
print(f"Columns: {list(df.columns)}")
print()

# --- Q1: Histogram of mean temperatures ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df['mean_temp'], bins=50, edgecolor='black', alpha=0.75, color='steelblue')
ax.set_xlabel('Mean temperature (°C)')
ax.set_ylabel('Number of buildings')
ax.set_title(f'Distribution of mean temperatures across {len(df)} buildings')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(HIST_OUT, dpi=120, bbox_inches='tight')
print(f"1. Histogram saved to {HIST_OUT}")
print()

# --- Q2: Average mean temperature ---
avg_mean = df['mean_temp'].mean()
print(f"2. Average mean temperature:               {avg_mean:.4f} °C")

# --- Q3: Average temperature std ---
avg_std = df['std_temp'].mean()
print(f"3. Average temperature std deviation:      {avg_std:.4f} °C")

# --- Q4: Buildings with >=50% area > 18°C ---
n_hot = (df['pct_above_18'] >= 50).sum()
print(f"4. Buildings with ≥50% of area above 18°C: {n_hot}  ({n_hot/len(df)*100:.2f}%)")

# --- Q5: Buildings with >=50% area < 15°C ---
n_cold = (df['pct_below_15'] >= 50).sum()
print(f"5. Buildings with ≥50% of area below 15°C: {n_cold}  ({n_cold/len(df)*100:.2f}%)")

# --- Bonus: distribution summary for the report ---
print()
print("--- Extra summary (for the report, optional) ---")
print(f"Mean temp:  min = {df['mean_temp'].min():.2f}, "
      f"median = {df['mean_temp'].median():.2f}, "
      f"max = {df['mean_temp'].max():.2f}")
print(f"Std temp:   min = {df['std_temp'].min():.2f}, "
      f"median = {df['std_temp'].median():.2f}, "
      f"max = {df['std_temp'].max():.2f}")