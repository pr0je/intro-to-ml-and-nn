# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Global plot settings
sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize':9
})

pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', '{:.3f}'.format)

# Structure and Characteristics of the Dataset

# shape
print(f"\n► SHAPE")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Total data points: {df.shape[0] * df.shape[1]:,}")

# Column names and raw datatypes:
print(f"  {'Column':<30} {'Dtype':<15} {'Non-Null Count'}")
for col in df.columns:
  non_null = df[col].notna().sum()
  print(f" {col:<30} {str(df[col].dtype):<15} {non_null:,}")

# Classify columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
object_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n► COLUMN CLASSIFICATION:")
print(f"  Numeric columns  ({len(numeric_cols)}): {numeric_cols}")
print(f"  Object  columns  ({len(object_cols)}) : {object_cols}")


# Categorical columns to 'category' dtypes.

cat_cols_to_convert = [
    'term',               # '36 months' or '60 months'
    'grade',              # A, B, C, D, E, F, G
    'sub_grade',          # A1 … G5
    'emp_length',         # '< 1 year', '1 year', …, '10+ years'
    'home_ownership',     # RENT, MORTGAGE, OWN, OTHER, NONE
    'verification_status',# Not Verified, Source Verified, Verified
    'loan_status',        # Fully Paid, Charged Off  ← TARGET
    'purpose',            # debt_consolidation, credit_card, …
    'initial_list_status',# w (whole) or f (fractional)
    'application_type',   # INDIVIDUAL or JOINT
]

# Only convert columns that actually exist in the dataframe
cat_cols_to_convert = [c for c in cat_cols_to_convert if c in df.columns]

print(f"\n  Converting {len(cat_cols_to_convert)} columns to 'category'...\n")
for col in cat_cols_to_convert:
    before_dtype = str(df[col].dtype)
    df[col] = df[col].astype('category')
    unique_vals = df[col].nunique()
    print(f"  ✅ {col:<25} {before_dtype:<12} → category  "
          f"({unique_vals} unique values)")

print(f"\n► Current dtypes after conversion:")
print(df.dtypes)

# Missing values detection
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column'        : missing_count.index,
    'Missing Count' : missing_count.values,
    'Missing %'     : missing_pct.values,
    'Dtypes'        : df.dtypes.values
}).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

print(f"\n Columns WITH missing values ({len(missing_df)} found):\n")
display(missing_df.reset_index(drop=True))

print(f"\n  Columns with ZERO missing values: "f"{(missing_count == 0).sum()} columns are complete.")

# Visual: Missing values bar chart
fig, ax = plt.subplots(figsize= (10,5))
colors = ['#E53935' if p > 5 else '#FB8C00' if p > 1 else '#43A047'
            for p in missing_df['Missing %']]

bars = ax.barh(missing_df['Column'], missing_df['Missing %'], color=colors, edgecolor='black', height=0.6)

# Add % labels on bars
for bar, pct in zip(bars, missing_df['Missing %']):
  ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontsize=9)

ax.set_xlabel('Missing Percentage (%)')
ax.set_title('Missing Values by Column\n(Red > 5%  | Orange 1-5%  | Green <1%)',fontweight='bold')
ax.axvline(5, color='red', linestyle = '--', lw=1.2, alpha=0.7,label='5% line')
ax.legend(fontsize=9)
ax.invert_yaxis() # highest missing at top
plt.tight_layout()
plt.savefig('missing_values.png', dpi=100, bbox_inches='tight')
plt.show()

# Statistical Summary 
# Numeric summary
print("\n► NUMERIC COLUMNS – Descriptive Statistics:")
display(df.describe().T.style.format('{:.2f}'))

# Categorical summary
print("\n► CATEGORICAL COLUMNS – Value Counts Summary:")
cat_summary = []

for col in cat_cols_to_convert:
  if col in df.columns:
    top_val = df[col].value_counts().index[0]
    top_freq = df[col].value_counts().iloc[0]
    top_pct = top_freq / len(df) * 100
    cat_summary.append({
        'Column': col,
        'Unique Values': df[col].nunique(),
        'Top Value' : str(top_val),
        'Top Count' : top_freq,
        'Top %': f'{top_pct:.1f}%'
    })

display(pd.DataFrame(cat_summary))

# Univariate Analysis - Continous Variable
from matplotlib import transforms
from IPython.core.pylabtools import figsize
# All continuous/numeric columns of interest.
continuous_cols = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc',
    'dti', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies'
]

continuous_cols = [c for c in continuous_cols if c in df.columns]

# Plot: Histogram with kde for all continous variables:

fig, axes = plt.subplots(4,3, figsize=(20,18))
fig.suptitle('Univariate Distributions - Continuous Variables\n(Histogram + KDE overlay)', fontsize=15, fontweight='bold', y=1.01)
axes = axes.flatten()

palette_colors = sns.color_palette('Set2', len(continuous_cols))

for i, col in enumerate(continuous_cols):
  data = df[col].dropna()
  ax   = axes[i]


  # Histogram
  ax.hist(data, bins=50, color=palette_colors[i],edgecolor='white', alpha=0.80,density=True)

  #KDE overlay
  try:
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(),300)
    ax.plot(x_range,kde(x_range), color='black',lw=1.8)
  except Exception:
    pass

  # Stats annotations:
  skew_val = data.skew()
  mean_val = data.mean()
  ax.axvline(mean_val, color='red', linestyle='--', lw=1.5, label='Mean')
  ax.axvline(data.median(),color='blue', linestyle='--', lw=1.5,label="Median")

  ax.set_title(col, fontsize=12, fontweight='bold')
  ax.set_xlabel('')
  ax.text(0.97,0.95, f'Skew: {skew_val:.2f}', transform=ax.transAxes, ha='right', va='top',fontsize=9, color='darkred',bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
  if i ==0:
    ax.legend(fontsize=8)

# Hide unused subplots:
for j in range(i + 1, len(axes)):
  axes[j].set_visible(False)


plt.tight_layout()
plt.savefig('univariate_continuous.png',dpi=100,bbox_inches='tight')
plt.show()

# Univariate Continours Variable (Boxplots)
# Boxplots to highlight outliers:
fig, axes = plt.subplots(3,4,figsize=(20,14))
fig.suptitle('Boxplots - Outliers Detection for Continuous Variables',fontsize=15,fontweight='bold',y=1.01)
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
  data = df[col].dropna()
  ax = axes[i]

  bp = ax.boxplot(data, vert=True, patch_artist=True,
                  boxprops=dict(facecolor=palette_colors[i], alpha=0.7),
                  medianprops=dict(color='red',linewidth=2),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5),
                  flierprops=dict(marker='o',markerfacecolor='gray',markersize=2,alpha=0.3))

  # IQR and outlier bounds
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  n_outilers = ((data < lower) | (data > upper)).sum()

  ax.set_title(col, fontsize=11, fontweight='bold')
  ax.set_xlabel('')
  ax.text(0.97,0.97,
          f'Outilers: {n_outilers:,}\n({n_outilers/len(data)*100:.1f}%)',
          transform=ax.transAxes, ha='right', va='top',fontsize=8,color='darkred',
          bbox=dict(boxstyle='round,pad=0.2',facecolor='lightyellow',alpha=0.8))

for j in range(i + 1, len(axes)):
  axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('univariate_continuous_boxplots.png',dpi=100,bbox_inches='tight')
plt.show()




