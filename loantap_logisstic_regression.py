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

# Univariat Analysis - Categorical Variables
cat_plot_cols = ['loan_status', 'grade', 'home_ownership',
                 'verification_status', 'purpose', 'term',
                 'initial_list_status', 'application_type', 'emp_length']

cat_plot_cols = [c for c in cat_plot_cols if c in df.columns]

fig, axes = plt.subplots(3,3,figsize=(22,18))
fig.suptitle('Univariate Analysis - Categorical Variables\n(Count plots with percentage labels)',fontsize=15,fontweight='bold',y=1.01)
axes = axes.flatten()

for i, col in enumerate(cat_plot_cols):
  ax = axes[i]
  order = df[col].value_counts().index.tolist()

  sns.countplot(data=df, x=col, order=order, ax=ax, palette='Set2',edgecolor='black', linewidth=0.5)

  ax.set_title(col, fontsize=12, fontweight='bold')
  ax.set_xlabel('')
  ax.tick_params(axis='x',rotation=35)

  # Add percentage labels above each bar
  total = len(df)
  for p in ax.patches:
    height = p.get_height()
    if height > 0:
      ax.annotate(f'{height/total*100:.1f}%',
                  xy=(p.get_x() + p.get_width() / 2,height),
                  xytext=(0,4), textcoords='offset points',
                  ha='center',va='bottom',fontsize=8,fontweight='bold')

for j in range(i+1, len(axes)):
  axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('univariate_categorical.png',dpi=110,bbox_inches='tight')
plt.show()

# Bivariate Analysis - Target vs Predictors.
# Grade vs Loan Status
fig, axes = plt.subplots(1,2,figsize=(18,6))
fig.suptitle('Bivariate: Grade vs Loan Status',fontsize=14,fontweight='bold')

# Count Plot
grade_status = df.groupby(['grade','loan_status'], observed=True).size().unstack(fill_value=0)
grade_status.plot(kind='bar',ax=axes[0],color=['#43A047', '#E53935'], edgecolor='black', width=0.7)
axes[0].set_title('Count of Loans by Grade & Status')
axes[0].set_xlabel('Grade')
axes[0].tick_params(axis='x',rotation=0)
axes[0].legend(title='Loan Status',fontsize=9)

# Stacked % plot
grade_pct = grade_status.div(grade_status.sum(axis=1), axis=0) * 100
grade_pct.plot(kind='bar',stacked=True,ax=axes[1],
               color=['#43A047', '#E53935'], edgecolor='black',width=0.7)
axes[1].set_title('Default Rate (%) by Grade')
axes[1].set_xlabel('Grade')
axes[1].tick_params(axis='x', rotation=0)
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Loan Status', fontsize=9)

# Add default % labels
charged_pct = grade_pct.get('Charged Off', grade_pct.iloc[:, -1])
for j, (grade, pct) in enumerate(charged_pct.items()):
    axes[1].text(j, grade_pct.iloc[j, 0] + pct / 2,
                 f'{pct:.0f}%', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('bivariate_grade_status.png', dpi=110, bbox_inches='tight')
plt.show()


# Interest Rate by Loan Status;

fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle('Bivariate: Interest Rate vs Loan Status',fontsize=14,fontweight='bold')

# KDE / Histogram overlay

for label, grp in df.groupby('loan_status',observed=True):
  axes[0].hist(grp['int_rate'].dropna(),bins=50,alpha=0.55,
               label=str(label),density=True,edgecolor='white')
axes[0].set_title('Interest Rate Distribution by Loan Status')
axes[0].set_xlabel('Interest Rate (%)')
axes[0].set_ylabel('Density')
axes[0].legend(fontsize=9)

# Boxplot
df.boxplot(column='int_rate', by='loan_status', ax=axes[1],
           showfliers=True,
           patch_artist=True,
           boxprops=dict(facecolor='#90CAF9',alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Interest Rate Distribution by Loan Status')
axes[1].set_xlabel('Loan Status')
axes[1].set_ylabel('Interest Rate (%)')
plt.sca(axes[1])
plt.title('Interest Rate by Loan Status')
plt.suptitle('')

plt.tight_layout()
plt.savefig('bivariate_int_rate_status.png', dpi=110, bbox_inches='tight')
plt.show()

# Mean int_rate by status
print(df.groupby('loan_status',observed=True)['int_rate'].agg(['mean','median']).round(2))


# Loadn Amount by Loan Status

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Bivariate: Loan Amount vs Loan Status', fontsize=14, fontweight='bold')

for label, grp in df.groupby('loan_status', observed=True):
    axes[0].hist(grp['loan_amnt'].dropna(), bins=40, alpha=0.55,
                 label=str(label), density=True, edgecolor='white')
axes[0].set_title('Loan Amount Distribution by Loan Status')
axes[0].set_xlabel('Loan Amount ($)')
axes[0].set_ylabel('Density')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
axes[0].legend(fontsize=9)

df.boxplot(column='loan_amnt', by='loan_status', ax=axes[1],
           showfliers=True, patch_artist=True,
           boxprops=dict(facecolor='#CE93D8', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Loan Amount by Loan Status')
axes[1].set_xlabel('Loan Status')
plt.sca(axes[1]); plt.title('Loan Amount by Loan Status'); plt.suptitle('')

plt.tight_layout()
plt.savefig('bivariate_loanamnt_status.png', dpi=110, bbox_inches='tight')
plt.show()

print("  Mean loan amount by loan_status:")
print(df.groupby('loan_status', observed=True)['loan_amnt'].agg(['mean', 'median']).round(0))


# DTI by Loan Status

fig, axes = plt.subplots(1, 2, figsize=(16,6))
fig.suptitle('Bivariate: DTI vs Loan Status', fontsize=14, fontweight='bold')

for label, grp in df.groupby('loan_status',observed=True):
  clean = grp['dti'].dropna()
  clean = clean[clean < 60]
  axes[0].hist(clean, bins=40, alpha = 0.55,
               label=str(label), density=True,edgecolor='white')
axes[0].set_title('DTI Distribution by Loan Status (capped at 60)')
axes[0].set_xlabel('DTI')
axes[0].legend(fontsize=9)

df[df['dti'] < 60].boxplot(column='dti', by='loan_status', ax=axes[1],
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor='#FFCC80', alpha=0.8),
                           medianprops=dict(color='red',linewidth=2))

axes[1].set_title('DTI Boxplot by Loan Status')
axes[1].set_xlabel('Loan Status')
plt.sca(axes[1]); plt.title('DTI by Loan Status'); plt.suptitle('')

plt.tight_layout()
plt.savefig('bivariate_dti_status.png', dpi=110, bbox_inches='tight')
plt.show()

# Annual Income by Loan Status:
income_cap = df['annual_inc'].quantile(0.99)
df_cap = df[df['annual_inc'] < income_cap].copy()

fig, axes = plt.subplots(1, 2, figsize=(16,6))
fig.suptitle('Bivariate: Annual Income vs Loan Status', fontsize= 14,
             fontweight='bold')

for label, grp in df_cap.groupby('loan_status',observed=True):
  axes[0].hist(grp['annual_inc'].dropna(), bins=40, alpha=0.55,
               label=str(label),density=True, edgecolor='white')
axes[0].set_title('Annual Income Distribution by Loan Status')
axes[0].set_xlabel('Annual Income ($) ')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
axes[0].legend(fontsize=9)

df_cap.boxplot(column='annual_inc', by='loan_status', ax=axes[1],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#80DEEA', alpha=0.8),
               medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Income Boxplot by Loan Status')
axes[1].set_xlabel('Loan Status')
plt.sca(axes[1]); plt.title('Annual Income by Loan Status'); plt.suptitle('')

plt.tight_layout()
plt.savefig('bivariate_income_status.png',dpi=110, bbox_inches='tight')
plt.show()

print("  Mean annual income by loan_status:")
print(df.groupby('loan_status', observed=True)['annual_inc'].agg(['mean', 'median']).round(0))

# Home Ownership vs Loan Status:
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Bivariate: Home Ownership vs Loan Status', fontsize=14, fontweight='bold')

#Home owner count
ho_count = df.groupby(['home_ownership', 'loan_status'], observed=True).size().unstack(fill_value=0)
ho_pct = ho_count.div(ho_count.sum(axis=1), axis=0) * 100

ho_count.plot(kind='bar', ax=axes[0], color=['#43A047','#E53935'],
              edgecolor='black', width=0.7)
axes[0].set_title('Count by Home Owenership & Loan Status')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Status', fontsize=9)

ho_pct.plot(kind='bar', stacked=True, ax=axes[1],
            color=['#43A047', '#E53935'], edgecolor='black', width=0.7)
axes[1].set_title('Default Rate (%) by Home Ownership')
axes[1].tick_params(axis='x', rotation=20)
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Status', fontsize=9)

plt.tight_layout()
plt.savefig('bivariate_homeown_status.png', dpi=110, bbox_inches='tight')
plt.show()

print("  Default rate (%) by home_ownership:")
choff_col = [c for c in ho_pct.columns if 'Charged' in str(c)]
if choff_col:
    print(ho_pct[choff_col[0]].sort_values(ascending=False).round(1).to_string())

# Loan Term vs Loan Status:
fig, axes = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle('Bivariate: Loan Term vs Loan Status', fontsize=14, fontweight='bold')

term_count = df.groupby(['term', 'loan_status'], observed=True).size().unstack(fill_value=0)
term_pct = term_count.div(term_count.sum(axis=1), axis=0) * 100

term_count.plot(kind='bar', ax=axes[0], color=['#43A047', '#E53935'],
                edgecolor='black', width=0.5)
axes[0].set_title('Count by term & Loan Status')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Status', fontsize=9)

term_pct.plot(kind='bar', stacked=True, ax=axes[1],
              color=['#43A047', '#E53935'], edgecolor='black', width=0.5)
axes[1].set_title('Default Rate (%) by Term')
axes[1].tick_params(axis='x', rotation=0)
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Status', fontsize=9)

# Add labels
choff_col2 = [c for c in term_pct.columns if 'Charged' in str(c)]
if choff_col2:
    for k, (term_val, pct) in enumerate(term_pct[choff_col2[0]].items()):
        axes[1].text(k, 100 - pct/2, f'{pct:.1f}%',
                     ha='center', va='center', fontsize=11,
                     fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('bivariate_term_status.png', dpi=110, bbox_inches='tight')
plt.show()

# Verification Status VS Loan Status.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bivariate: Verification Status vs Loan Status', fontsize=14, fontweight='bold')

vs_count = df.groupby(['verification_status', 'loan_status'], observed=True).size().unstack(fill_value=0)
vs_pct = vs_count.div(vs_count.sum(axis=1), axis=0) * 100

vs_count.plot(kind='bar', ax=axes[0], color=['#43A047', '#E53935'],
              edgecolor='black', width=0.6)
axes[0].set_title('Count by Verification & Loan Status')
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(title='Status', fontsize=9)

vs_pct.plot(kind='bar', stacked=True, ax=axes[1],
            color=['#43A047', '#E53935'], edgecolor='black', width=0.6)
axes[1].set_title('Default Rate (%) by Verification Status')
axes[1].tick_params(axis='x', rotation=15)
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Status', fontsize=9)

plt.tight_layout()
plt.savefig('bivariate_verification_status.png', dpi=110, bbox_inches='tight')
plt.show()

choff_col3 = [c for c in vs_pct.columns if 'Charged' in str(c)]
if choff_col3:
    print("  Default rate (%) by verification_status:")
    print(vs_pct[choff_col3[0]].round(1).to_string())

# Bivariate - Continuous vs Continuous (Scatter + Corr):
# Scatter: Loan_amnt vs installment (the high-correlation pair):
fig, axes = plt.subplots(1, 3, figsize=(20,6))
fig.suptitle('Bivariate Scatter Plots - Key Continuous Pairs',
             fontsize=14, fontweight='bold')

# Sample for speed
sample = df.sample(5000, random_state=42)
status_colors = {'Full Paid': '#43A047', 'Charged Off': '#E53935'}

for label, grp in sample.groupby('loan_status', observed=True):
  col_color = status_colors.get(str(label), 'gray')
  axes[0].scatter(grp['loan_amnt'], grp['installment'],
                  alpha=0.3, s=8, label=str(label), color=col_color)
  axes[0].set_xlabel('Loan Amount ($)')
axes[0].set_ylabel('Installment ($)')
r0 = df[['loan_amnt', 'installment']].corr().iloc[0, 1]
axes[0].set_title(f'loan_amnt vs installment\n(r = {r0:.3f})')
axes[0].legend(fontsize=8)

for label, grp in sample.groupby('loan_status', observed=True):
    col_color = status_colors.get(str(label), 'gray')
    axes[1].scatter(grp['annual_inc'].clip(upper=df['annual_inc'].quantile(0.99)),
                    grp['loan_amnt'],
                    alpha=0.3, s=8, label=str(label), color=col_color)
axes[1].set_xlabel('Annual Income ($)')
axes[1].set_ylabel('Loan Amount ($)')
r1 = df[['annual_inc', 'loan_amnt']].corr().iloc[0, 1]
axes[1].set_title(f'annual_inc vs loan_amnt\n(r = {r1:.3f})')
axes[1].legend(fontsize=8)

for label, grp in sample.groupby('loan_status', observed=True):
    col_color = status_colors.get(str(label), 'gray')
    axes[2].scatter(grp['int_rate'], grp['dti'],
                    alpha=0.3, s=8, label=str(label), color=col_color)
axes[2].set_xlabel('Interest Rate (%)')
axes[2].set_ylabel('DTI')
r2 = df[['int_rate', 'dti']].corr().iloc[0, 1]
axes[2].set_title(f'int_rate vs dti\n(r = {r2:.3f})')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('bivariate_scatter.png', dpi=110, bbox_inches='tight')
plt.show()

# Full  Correlation Heatmap
corr_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc',
             'dti', 'open_acc', 'pub_rec', 'revol_bal',
             'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show only lower triangle

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='white',
    annot_kws={'size': 8},
    ax=ax,
    square=True
)

ax.set_title('Correlation Heatmap – Numeric Features\n(Lower triangle only)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=110, bbox_inches='tight')
plt.show()

# Print strong correlations
print("\n► Pairs with |r| > 0.40 (potential multicollinearity):")
print(f"  {'Feature 1':<25} {'Feature 2':<25} {'r':>8}")
print(f"  {'-'*60}")
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.40:
            print(f"  {corr_matrix.columns[i]:<25} {corr_matrix.columns[j]:<25} {r:>8.3f}")

# Top Job Titles Analysis:
if 'emp_title' in df.columns:
  # clean and count
  top_jobs = (
      df['emp_title'].astype(str)
      .str.strip()
      .str.title()
      .replace('Unknown',np.nan)
      .dropna()
      .value_counts()
      .head(10)
  )

  fig, ax = plt.subplots(figsize=(12, 7))

  colors_jobs = sns.color_palette('viridis', len(top_jobs))

  bars = ax.barh(top_jobs.index[::-1], top_jobs.values[::-1],
                   color=colors_jobs, edgecolor='black', height=0.7)

  ax.set_title('Top 15 Job Titles Among LoanTap Borrowers',
                 fontsize=13, fontweight='bold')

  ax.set_xlabel('Number of Borrowers')

  for bar, val in zip(bars, top_jobs.values[::-1]):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=8)

  plt.tight_layout()
  plt.savefig('top_job_titles.png', dpi=110, bbox_inches='tight')
  plt.show()

  print("\n  Top 5 Job Titles:")
  for rank, (title, count) in enumerate(top_jobs.head(5).items(), 1):
      print(f"  {rank}. {title:<30} {count:>8,} borrowers")

# Data Preprocessing

# ── Snapshot: state of df before ANY preprocessing
print(f"  📸 BEFORE PREPROCESSING SNAPSHOT:")
print(f"  Rows      : {df.shape[0]:,}")
print(f"  Columns   : {df.shape[1]}")
print(f"  Missing   : {df.isnull().sum().sum():,} total missing cells")
print(f"  Duplicates: will check in Step 2a")

# DUPLICATE VALUE CHECK
# Full row duplicates
n_full_dups = df.duplicated().sum()
print(f"\n► Full row duplicates (all 27 columns identical): {n_full_dups:,}")

# Show duplicate rows if any exist
if n_full_dups > 0:
    print(f"\n  Sample of duplicate rows:")
    display(df[df.duplicated(keep=False)].head(6))
else:
    print("  ✅ No full-row duplicates found.")

# remove duplicates
rows_before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
rows_after = len(df)

print(f"\n► Rows before drop : {rows_before:,}")
print(f"► Rows after drop  : {rows_after:,}")
print(f"► Rows removed     : {rows_before - rows_after:,}")

# Check duplicates on key identifying columns (not address/title which vary)
key_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc',
            'dti', 'open_acc', 'revol_bal', 'revol_util']
key_cols = [c for c in key_cols if c in df.columns]
n_key_dups = df.duplicated(subset=key_cols).sum()

print(f"\n► Near-duplicates (same on {len(key_cols)} numeric key columns): {n_key_dups:,}")
print(f"  Note: These are different borrowers with identical profiles — RETAIN.")

# **MISSING VALUE TREATMENT - Analysis:
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
misssing_df = (
    pd.DataFrame({'Count': missing, 'Pct (%)': missing_pct})
    .query('Count > 0')
    .sort_values('Pct (%)', ascending=False)
)
print('Columns with missing values (Before Treatment): ')
display(missing_df)

# Missing value treatment
# mort_acc --> Median impuatation
mort_median = df['mort_acc'].median()
n_filled = df['mort_acc'].isnull().sum()
df['mort_acc'].fillna(mort_median, inplace=True)
print(f"  ✅ mort_acc          : filled {n_filled:,} NaNs with median = {mort_median:.1f}")

# emp_title -> 'Unknown'
if 'emp_title' in df.columns:
    n_filled = df['emp_title'].isnull().sum()
    df['emp_title'].fillna('Unknown', inplace=True)
    print(f"  ✅ emp_title         : filled {n_filled:,} NaNs with 'Unknown'")

# emp_length -> convert to numeric first, then fill 0
# convert before filling so we get numeric 0
df['emp_length'] = (
    df['emp_length']
    .astype(str)
    .str.extract(r'(\d+)')[0]   # extract first digit(s)
    .astype(float)
)
n_filled = df['emp_length'].isnull().sum()
df['emp_length'].fillna(0, inplace=True)
print(f"  ✅ emp_length         : converted to numeric + filled {n_filled:,} NaNs with 0")

# title -> 'Unknown'
if 'title' in df.columns:
    n_filled = df['title'].isnull().sum()
    df['title'].fillna('Unknown', inplace=True)
    print(f"  ✅ title             : filled {n_filled:,} NaNs with 'Unknown'")

#  pub_rec_bankruptcies -> 0
n_filled = df['pub_rec_bankruptcies'].isnull().sum()
df['pub_rec_bankruptcies'].fillna(0, inplace=True)
print(f"  ✅ pub_rec_bankruptcies: filled {n_filled:,} NaNs with 0")

#  revol_util -> Median imputation
revol_median = df['revol_util'].median()
n_filled = df['revol_util'].isnull().sum()
df['revol_util'].fillna(revol_median, inplace=True)
print(f"  ✅ revol_util         : filled {n_filled:,} NaNs with median = {revol_median:.1f}")
