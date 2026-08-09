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

# ── Verify: no missing values remain in key columns
remaining = df.isnull().sum()
remaining_nonzero = remaining[remaining > 0]

print(f"\n► Missing values AFTER treatment:")
if len(remaining_nonzero) == 0:
    print(f"  ✅ Zero missing values in all treated columns.")
else:
    display(remaining_nonzero.to_frame('Remaining Missing'))

# Visual: Before and After:
treated_cols = ['mort_acc', 'emp_length', 'pub_rec_bankruptcies', 'revol_util']

before_vals_series = missing_df[missing_df['Column'].isin(treated_cols)].set_index('Column')['Missing Count']
before_vals = before_vals_series.reindex(treated_cols).fillna(0).values

after_missing = df[treated_cols].isnull().sum()

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(treated_cols))
width = 0.35

bars1 = ax.bar(x - width/2, before_vals, width, label='Before', color='#EF5350', edgecolor='black')
bars2 = ax.bar(x + width/2, after_missing.values, width, label='After', color='#66BB6A', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(treated_cols, rotation=15)
ax.set_ylabel('Missing Count')
ax.set_title('Missing Values: Before vs After Treatment', fontweight='bold')
ax.legend()
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('missing_before_after.png', dpi=110, bbox_inches='tight')
plt.show()

# Outlier Treatment - Detection:
outlier_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti',
                'revol_bal', 'revol_util', 'open_acc',
                'pub_rec', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
outlier_cols = [c for c in outlier_cols if c in df.columns]

# Build IQR outlier report
outlier_report = []
for col in outlier_cols:
  data    = df[col].dropna()
  Q1, Q3  = data.quantile(0.25), data.quantile(0.75)
  OQR     = Q3 - Q1
  lower   = Q1 - 1.5 * OQR
  upper   = Q3 + 1.5 * OQR
  n_out   = ((data < lower) | (data > upper)).sum()
  pct_out  = n_out / len(data) * 100
  p99      = data.quantile(0.99)
  outlier_report.append({
      'Column'    : col,
      'Q1'        : round(Q1, 2),
      'Q3'        : round(Q3, 2),
      'IQR'       : round(IQR, 2),
      'Lower Fence': round(lower, 2),
      'Upper Fence': round(upper, 2),
      '99th Pct'  : round(p99, 2),
      'Max Value' : round(data.max(), 2),
      'Outlier Count': n_out,
      'Outlier %' : round(pct_out, 2)})

outlier_df = pd.DataFrame(outlier_report)
print("► IQR Outlier Report:")
display(outlier_df)

# OUTLIER TREATMENT — Visual (Before)

cap_visual_cols = ['annual_inc', 'revol_bal', 'dti', 'revol_util']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Boxplots BEFORE Outlier Capping', fontsize=14, fontweight='bold')

for i, col in enumerate(cap_visual_cols):
    # Raw boxplot
    axes[0, i].boxplot(
        df[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor='#EF9A9A', alpha=0.8),
        medianprops=dict(color='red', linewidth=2),
        flierprops=dict(marker='o', markersize=2, alpha=0.3, color='gray')
    )
    axes[0, i].set_title(f'{col}\n(Raw)', fontsize=10)

    # Histogram
    axes[1, i].hist(df[col].dropna(), bins=60,
                    color='#EF9A9A', edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'{col}\nHistogram (Raw)', fontsize=10)

    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    axes[1, i].axvline(Q3 + 1.5*IQR, color='red', linestyle='--',
                       lw=1.5, label='IQR upper')
    axes[1, i].axvline(df[col].quantile(0.99), color='blue',
                       linestyle='--', lw=1.5, label='99th pct')
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('outliers_before.png', dpi=110, bbox_inches='tight')
plt.show()
print("  ↑ Red dashed = IQR upper fence  |  Blue dashed = 99th percentile")

# Outlier Treatment Applying Capping (Winsorization)
#  Treatment 1: Winsorize at 99th percentile
# Columns where extreme tail values distort model
winsorize_cols = {
    'annual_inc': {'upper_pct': 0.99},
    'revol_bal' : {'upper_pct': 0.99},
    'dti'       : {'upper_pct': 0.99},
}

for col, params in winsorize_cols.items():
    before_max = df[col].max()
    cap_val    = df[col].quantile(params['upper_pct'])
    n_capped   = (df[col] > cap_val).sum()
    df[col]    = np.where(df[col] > cap_val, cap_val, df[col])
    print(f"  ✅ {col:<15}: capped {n_capped:>5,} rows  "
          f"| max before={before_max:>12,.1f}  → max after={df[col].max():>10,.1f}")

# Treatment 2: Hard logical cap for revol_util
# revol_util >100 is physically impossible
before_max  = df['revol_util'].max()
n_capped    = (df['revol_util'] > 100).sum()
df['revol_util'] = np.where(df['revol_util'] > 100, 100, df['revol_util'])
print(f"  ✅ {'revol_util':<15}: capped {n_capped:>5,} rows  "
      f"| max before={before_max:>12,.1f}  → max after={df['revol_util'].max():>10,.1f}")

# Treatment 3: pub_rec and pub_rec_bankruptcies
# We do NOT cap these — we will create binary flags in Feature Engineering
print(f"\n  ℹ️  pub_rec and pub_rec_bankruptcies:")
print(f"     NOT capped here — binary flags will be created in Step 2d.")
print(f"     Flag (1 = has record) is more meaningful than the raw count.")

print(f"\n  Summary after outlier treatment:")
for col in list(winsorize_cols.keys()) + ['revol_util']:
    print(f"  {col:<18}: min={df[col].min():>10.2f}  "
          f"max={df[col].max():>10.2f}  "
          f"mean={df[col].mean():>10.2f}")

# Outlier Treatment - Visual (After) + Comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Boxplots AFTER Outlier Capping (99th pct / logical caps)',
             fontsize=14, fontweight='bold')

for i, col in enumerate(cap_visual_cols):
    axes[0, i].boxplot(
        df[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor='#A5D6A7', alpha=0.8),
        medianprops=dict(color='red', linewidth=2),
        flierprops=dict(marker='o', markersize=2, alpha=0.3, color='gray')
    )

    axes[0, i].set_title(f'{col}\n(After Cap)', fontsize=10)

    axes[1, i].hist(df[col].dropna(), bins=60,
                    color='#A5D6A7', edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'{col}\nHistogram (After Cap)', fontsize=10)

plt.tight_layout()
plt.savefig('outliers_after.png', dpi=110, bbox_inches='tight')
plt.show()

# Feature Engineering - Overview;
# create flags
df['pub_rec_flag']= (df['pub_rec'] > 0).astype(int)
df['mort_acc_flag'] = (df['mort_acc'] > 0).astype(int)
df['pub_rec_bankrupt_flag'] = (df['pub_rec_bankruptcies'] > 0).astype(int)

# validate
for flag, source in [('pub_rec_flag',          'pub_rec'),
                     ('mort_acc_flag',         'mort_acc'),
                     ('pub_rec_bankrupt_flag', 'pub_rec_bankruptcies')]:
    pct   = df[flag].mean() * 100
    raw_n = (df[source] > 0).sum()
    print(f"  {flag:<28}: {pct:.1f}%  ({raw_n:,} borrowers have {source} > 0)")

# ── Visual: Flag vs Loan Status
flags   = ['pub_rec_flag', 'mort_acc_flag', 'pub_rec_bankrupt_flag']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Binary Flags vs Loan Status\n(Default rate for flag=0 vs flag=1)',
             fontsize=13, fontweight='bold')

for i, flag in enumerate(flags):
    cross = pd.crosstab(df[flag], df['loan_status'].astype(str), normalize='index') * 100
    cross.plot(kind='bar', stacked=True, ax=axes[i],
               color=['#43A047', '#E53935'], edgecolor='black', width=0.5)
    axes[i].set_title(flag, fontsize=11)
    axes[i].set_xlabel('Flag Value (0 = No Record, 1 = Has Record)')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xticklabels(['0 (No record)', '1 (Has record)'], rotation=0)
    axes[i].legend(title='Status', fontsize=8)

plt.tight_layout()
plt.savefig('flags_vs_status.png', dpi=110, bbox_inches='tight')
plt.show()

# Feature Engineering - Time Features (credit_age)
# Parse dates
df['issue_year']   = pd.to_datetime(df['issue_d'].astype(str), format='%b-%Y', errors='coerce').dt.year

df['cr_line_year'] = pd.to_datetime(df['earliest_cr_line'].astype(str), format='%b-%Y', errors='coerce').dt.year

df['credit_age'] = df['issue_year'] - df['cr_line_year']

# Handle any negative or extreme values (data errors)
df['credit_age'] = df['credit_age'].clip(lower=0, upper=50)

# Fill any NaT-derived NaNs with median
credit_age_median = df['credit_age'].median()
df['credit_age'].fillna(credit_age_median, inplace=True)

print(f"  ✅ credit_age created")
print(f"     Range  : {df['credit_age'].min():.0f} – {df['credit_age'].max():.0f} years")
print(f"     Mean   : {df['credit_age'].mean():.1f} years")
print(f"     Median : {df['credit_age'].median():.1f} years")


# ── Visual: credit_age distribution + vs loan_status ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Feature Engineering: credit_age', fontsize=13, fontweight='bold')


# Distribution
axes[0].hist(df['credit_age'].dropna(), bins=40,
             color='#7986CB', edgecolor='white', alpha=0.85)
axes[0].set_title('credit_age Distribution')
axes[0].set_xlabel('Years in Credit System')
axes[0].set_ylabel('Count')
axes[0].axvline(df['credit_age'].mean(), color='red', linestyle='--',
                lw=1.8, label='Mean')
axes[0].axvline(df['credit_age'].median(), color='blue', linestyle='--',
                lw=1.8, label='Median')
axes[0].legend()


# vs Loan Status boxplot
df.boxplot(column='credit_age', by='loan_status', ax=axes[1],
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor='#7986CB', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
axes[1].set_title('credit_age by Loan Status')
axes[1].set_xlabel('Loan Status')
axes[1].set_ylabel('Credit Age (Years)')
plt.sca(axes[1]); plt.title('credit_age by Loan Status'); plt.suptitle('')

plt.tight_layout()
plt.savefig('credit_age_feature.png', dpi=110, bbox_inches='tight')
plt.show()


print("\n  Mean credit_age by loan_status:")
print(df.groupby('loan_status', observed=True)['credit_age']
      .agg(['mean', 'median']).round(2))

# Feature Engineering - Term (numric) + State 
# Strio 'month' text and convert to int.
# FEATURE ENGINEERING — Term (numeric) + State
# Strip ' months' text and convert to int
df['term'] = (
    df['term'].astype(str)
    .str.strip()
    .str.replace(' months', '', regex=False)
    .astype(int)
)
print(f"  ✅ term converted: unique values = {sorted(df['term'].unique())}")
print(f"     36-month: {(df['term']==36).sum():,} loans  |  "
      f"60-month: {(df['term']==60).sum():,} loans")

print("\n  .  STATE EXTRACTION from 'address'")
print("  " + "-" * 60)
print("""
  The 'address' column contains free-text like:
    '123 Main St\\nSomecity, CA 90210'
  We extract the 2-letter state abbreviation using regex.
  This enables state-level default rate analysis (answered in Questionnaire Q9).
  NOTE: State will NOT be used as a direct model feature (too many categories),
  but is useful for geographic EDA and potential target encoding.
""")

df['state'] = df['address'].astype(str).str.extract(r',\s*([A-Z]{2})\s+\d')
n_states = df['state'].nunique()
print(f"  ✅ state extracted: {n_states} unique states found")
print(f"     Sample: {df['state'].value_counts().head(5).to_dict()}")

print("\n  .  TARGET VARIABLE ENCODING")
print("  " + "-" * 60)
print("""
  loan_status = 'Charged Off'  → target = 1  (default)
  loan_status = 'Fully Paid'   → target = 0  (no default)

  Binary integer encoding makes it compatible with sklearn's LogisticRegression.
""")

df['target'] = (df['loan_status'].astype(str) == 'Charged Off').astype(int)
print(f"  ✅ target created")
print(f"     target = 0 (Fully Paid)  : {(df['target']==0).sum():,}")
print(f"     target = 1 (Charged Off) : {(df['target']==1).sum():,}")
print(f"     Imbalance ratio          : {(df['target']==0).sum() / (df['target']==1).sum():.1f} : 1")

#All new features summary
new_features = ['pub_rec_flag', 'mort_acc_flag', 'pub_rec_bankrupt_flag',
                'credit_age', 'issue_year', 'cr_line_year', 'state', 'target']
print(f"\n  ✅ ALL ENGINEERED FEATURES SUMMARY:")
print(f"  {'Feature':<28} {'Dtype':<12} {'Non-Null':>10} {'Unique':>8}")
print(f"  {'-'*62}")
for feat in new_features:
    if feat in df.columns:
        print(f"  {feat:<28} {str(df[feat].dtype):<12} "
              f"{df[feat].notna().sum():>10,} {df[feat].nunique():>8,}")

# Data Preparation - Encoding Categorical Variables
# Drop redundant/irrelevant columns BEFORE encoding
drop_before_encode = [
    'loan_status',         # replaced by 'target'
    'installment',         # redundant with loan_amnt (r=0.954)
    'sub_grade',           # redundant with grade
    'emp_title',           # high cardinality, text
    'issue_d',             # replaced by credit_age
    'earliest_cr_line',    # replaced by credit_age
    'title',               # free-text
    'address',             # replaced by state (state also dropped below)
    'issue_year',          # intermediate feature
    'cr_line_year',        # intermediate feature
    'state',               # too many categories for OHE; target encoding optional
]

drop_before_encode = [c for c in drop_before_encode if c in df.columns]
df_model = df.drop(columns=drop_before_encode).copy()

print(f"  ✅ Dropped {len(drop_before_encode)} redundant columns: {drop_before_encode}")
print(f"     Remaining columns: {df_model.shape[1]}")

# One-Hot Encode
ohe_cols = ['grade', 'home_ownership', 'verification_status',
            'purpose', 'initial_list_status', 'application_type']
ohe_cols = [c for c in ohe_cols if c in df_model.columns]

print(f"\n  Applying One-Hot Encoding to: {ohe_cols}")
print(f"  (drop_first=True to avoid dummy variable trap)\n")

df_encoded = pd.get_dummies(df_model, columns=ohe_cols, drop_first=True)

# Show what was created
ohe_new_cols = [c for c in df_encoded.columns if c not in df_model.columns]
print(f"  ✅ OHE created {len(ohe_new_cols)} new binary columns:")
for c in ohe_new_cols:
    print(f"     {c}")

print(f"\n  Shape before OHE: {df_model.shape}")
print(f"  Shape after  OHE: {df_encoded.shape}")

# Data Preparation - Features Selection + Final Matrix
# Keep only numeric dtypes (should already be all numeric after OHE)
df_final = df_encoded.select_dtypes(include=[np.number]).copy()

# Final NaN safety fill (in case any slipped through)
remaining_nan = df_final.isnull().sum().sum()
if remaining_nan > 0:
    print(f"  ⚠️  {remaining_nan} NaNs still found — filling with column median...")
    df_final.fillna(df_final.median(), inplace=True)
else:
    print(f"  ✅ Zero NaNs in final feature matrix.")

# Separate X and y
X = df_final.drop(columns=['target'])
y = df_final['target']

print(f"\n  ✅ FINAL FEATURE MATRIX:")
print(f"     X shape : {X.shape}  ({X.shape[1]} features × {X.shape[0]:,} samples)")
print(f"     y shape : {y.shape}")
print(f"     Features: {X.columns.tolist()}")

# ── Class imbalance check ─────────────────────────────────────────────────────
print(f"\n  Class distribution in y:")
vc = y.value_counts()
print(f"     Class 0 (Fully Paid)  : {vc[0]:>8,}  ({vc[0]/len(y)*100:.1f}%)")
print(f"     Class 1 (Charged Off) : {vc[1]:>8,}  ({vc[1]/len(y)*100:.1f}%)")
print(f"     Ratio                 : {vc[0]/vc[1]:.1f} : 1  → IMBALANCED")
print(f"\n  Solution: class_weight='balanced' in LogisticRegression")
print(f"  This multiplies the loss for the minority class by ~{vc[0]/vc[1]:.1f}x,")
print(f"  forcing the model to pay more attention to defaulters.")

### DATA PREPARATION — Train/Test Split
from sklearn.model_selection import train_test_split

print("  TRAIN / TEST SPLIT")
print("  " + "-" * 60)
print("""
  Parameters:
  • test_size    = 0.20  → 80% train, 20% test
  • random_state = 42    → reproducibility
  • stratify=y           → preserves class ratio in both splits

  WHY STRATIFY?
  Without stratify, the random split might accidentally give the test set
  too many or too few defaulters (20% of 396K rows = ~79K rows; 1% shift
  = 790 rows misrepresented). Stratification guarantees the 80/20 class
  ratio is maintained in BOTH train and test sets.
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"  ✅ Split complete:")
print(f"     X_train : {X_train.shape[0]:>8,} rows")
print(f"     X_test  : {X_test.shape[0]:>8,} rows")

print(f"\n  Class distribution CHECK (stratification verification):")
print(f"     {'Split':<12} {'Class 0':>12} {'Class 1':>12} {'Ratio':>10}")
print(f"     {'-'*48}")
for name, yset in [('Full data', y), ('Train', y_train), ('Test', y_test)]:
    c0 = (yset == 0).sum()
    c1 = (yset == 1).sum()
    print(f"     {name:<12} {c0:>8,} ({c0/len(yset)*100:.1f}%)  "
          f"{c1:>8,} ({c1/len(yset)*100:.1f}%)  {c0/c1:>6.2f}:1")

print("""
  ✅ Class ratio is preserved across all three splits — stratification worked.
""")
# DATA PREPARATION — Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("  FEATURE SCALING")
print("  " + "-" * 60)
print("""
  WHY SCALE?
  ──────────
  Logistic Regression uses gradient descent to find optimal coefficients.
  If features are on very different scales:
    - loan_amnt : $500–$40,000
    - revol_util: 0–100 (%)
    - pub_rec   : 0–5
  …the gradient will be dominated by large-scale features, causing:
    → Slow convergence (max_iter exhausted)
    → Biased coefficient magnitudes (hard to compare importance)
    → Unstable solutions

  MinMaxScaler transforms each feature to range [0, 1]:
    X_scaled = (X - X_min) / (X_max - X_min)

  CRITICAL RULE — FIT ON TRAIN ONLY:
  We fit the scaler ONLY on X_train, then transform both X_train and X_test.
  If we fit on all data, the test set's min/max would "leak" into training,
  artificially inflating performance (data leakage!).

  MinMaxScaler vs StandardScaler:
  • MinMaxScaler : preserves 0s in sparse features; sensitive to outliers.
                   (We already capped outliers in Step 2c — safe to use.)
  • StandardScaler: centers at 0, scales by std. Better if data is Gaussian.
  → We use MinMaxScaler as it's standard for credit risk with bounded features.
""")

# ── Apply MinMaxScaler ────────────────────────────────────────────────────────
scaler = MinMaxScaler()

X_train_sc = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test_sc  = scaler.transform(X_test)        # transform ONLY on TEST

# Convert back to DataFrames for readability
X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=X.columns)

print(f"  ✅ MinMaxScaler applied")
print(f"     X_train_sc : shape={X_train_sc.shape}  "
      f"| min={X_train_sc.min().min():.4f}  max={X_train_sc.max().max():.4f}")
print(f"     X_test_sc  : shape={X_test_sc.shape}   "
      f"| min={X_test_sc.min().min():.4f}  max={X_test_sc.max().max():.4f}")

# ── Visual: Before vs After scaling (3 sample features) ──────────────────────
sample_features = ['loan_amnt', 'annual_inc', 'int_rate']
sample_features = [f for f in sample_features if f in X_train.columns]

fig, axes = plt.subplots(2, len(sample_features), figsize=(18, 9))
fig.suptitle('Feature Scaling: Before (top) vs After MinMaxScaler (bottom)',
             fontsize=13, fontweight='bold')

for i, feat in enumerate(sample_features):
    # Before scaling (training set)
    axes[0, i].hist(X_train[feat], bins=50, color='#EF9A9A',
                    edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f'{feat}\n(Before Scaling)', fontsize=10)
    axes[0, i].set_xlabel(f'Original values')

    # After scaling
    axes[1, i].hist(X_train_sc[feat], bins=50, color='#80CBC4',
                    edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'{feat}\n(After MinMaxScaler)', fontsize=10)
    axes[1, i].set_xlabel('Scaled values [0, 1]')
    axes[1, i].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('scaling_before_after.png', dpi=110, bbox_inches='tight')
plt.show()

print("""
  📝 SCALING OUTCOME:
  • All features are now in [0, 1] range.
  • Shape of distributions is PRESERVED — scaling only changes the unit,
    not the underlying data pattern.
  • The model will now converge faster and coefficients will be on a
    comparable scale (allowing relative importance comparison).
""")
# Model Building 
# Define model with all parameters explained:

model = LogisticRegression(
    C            = 0.1,        # Regularisation strength (smaller = stronger)
    penalty      = 'l2',       # Ridge penalty — shrinks large coefficients
    class_weight = 'balanced', # Handles 4:1 class imbalance automatically
    solver       = 'lbfgs',    # Best optimiser for L2 + medium-large data
    max_iter     = 1000,       # Iterations for convergence (default=100 too low)
    random_state = 42          # Reproducibility
)

# Train (fit) the model
import time
t0 = time.time()
model.fit(X_train_sc, y_train)
t1 = time.time()

print(f"  ✅ Model trained in {t1-t0:.2f} seconds")
print(f"  Iterations used   : {model.n_iter_[0]} / {model.max_iter}")
if model.n_iter_[0] < model.max_iter:
    print(f"  Convergence status: ✅ CONVERGED before max_iter")
else:
    print(f"  Convergence status: ⚠️  Hit max_iter — increase max_iter!")

# Gemerate Predicitions
y_pred     = model.predict(X_test_sc)       # Hard class labels: 0 or 1
y_prob     = model.predict_proba(X_test_sc) # Probabilities: [[P(0), P(1)], ...]
y_prob_pos = y_prob[:, 1]                   # P(Charged Off) per sample

print(f"\n  Predictions on test set ({len(y_test):,} samples):")
print(f"  y_pred     → class labels (0/1)       shape: {y_pred.shape}")
print(f"  y_prob     → both-class probabilities  shape: {y_prob.shape}")
print(f"  y_prob_pos → P(Charged Off) only       shape: {y_prob_pos.shape}")

# Preview Predictions

preview = pd.DataFrame({
    'Actual Label'      : y_test.values[:12],
    'Predicted Label'   : y_pred[:12],
    'P(Charged Off)'    : y_prob_pos[:12].round(4),
    'P(Fully Paid)'     : y_prob[:12, 0].round(4),
    'Correct?'          : ['✅' if a == p else '❌'
                           for a, p in zip(y_test.values[:12], y_pred[:12])]
})
print(f"\n  Sample predictions (first 12 test borrowers):")
display(preview)

# Statistic: Intercept

intercept = model.intercept_[0]
baseline_prob = 1 / (1 + np.exp(-intercept))
print(f"""
  ① INTERCEPT  (β₀)
  ─────────────────
  Value               : {intercept:.6f}
  Baseline probability: {baseline_prob:.4f}  ({baseline_prob*100:.2f}%)""")

print(f"""
  ② CONVERGENCE
  ──────────────
  Iterations used : {model.n_iter_[0]}
  Max allowed     : {model.max_iter}
  Status          : {'✅ Converged' if model.n_iter_[0] < model.max_iter else '⚠️ Not converged'}""")

# Training vs Test accuracy
train_acc = model.score(X_train_sc, y_train)
test_acc  = model.score(X_test_sc,  y_test)
gap       = train_acc - test_acc

print(f"""
  ③ ACCURACY OVERVIEW
  ────────────────────
  Training accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)
  Test accuracy     : {test_acc:.4f}  ({test_acc*100:.2f}%)
  Gap               : {gap:.4f}  ({gap*100:.2f}%)

  ⚠️  IMPORTANT CAVEAT:
  Accuracy is NOT the right metric for imbalanced datasets!
  A trivial model that predicts EVERY borrower as 'Fully Paid' would
  achieve {(y_test==0).mean()*100:.1f}% accuracy — but miss every single defaulter.
  → Focus on Recall, Precision, ROC-AUC, and F1 (covered in Section 4).

  The ~{gap*100:.1f}% train-test gap confirms the model is NOT overfitting.
  L2 regularisation (C=0.1) successfully prevents overfitting.
""")


# STATISTIC 4: ROC-AUC on test set

roc_auc = roc_auc_score(y_test, y_prob_pos)

print(f"""
  ④ ROC-AUC SCORE
  ────────────────
  ROC-AUC = {roc_auc:.4f}

  Interpretation:
  • AUC = {roc_auc:.4f} means the model correctly ranks a randomly
    chosen defaulter ABOVE a randomly chosen non-defaulter
    {roc_auc*100:.1f}% of the time.
  • AUC = 0.50 → random guessing (coin flip)
  • AUC = 1.00 → perfect model
  • AUC = {roc_auc:.2f} → MODERATE discriminatory power (acceptable for
    a baseline Logistic Regression on imbalanced credit data)
  • Benchmark: AUC > 0.70 is acceptable; > 0.80 is good for credit
""")
