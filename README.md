# Loantap: Logisttic Regression


## Loan Default Prediction using Logistic Regression

###  Project Overview

LoanTap is a fintech platform that provides instant personal loans.
This project builds a **machine learning model** to predict whether a borrower will:

* ✅ Fully repay the loan
* ❌ Default (Charged Off)

The goal is to **minimize financial loss (NPAs)** by identifying high-risk borrowers.

---

###  Business Objective

LoanTap faces two key risks:

1. **High Risk (Critical)** → Approving a bad borrower → Loan default → Financial loss
2. **Low Risk** → Rejecting a good borrower → Missed revenue

👉 Since Risk 1 is more severe, the model prioritizes:

* **Recall (detect defaulters)** over accuracy

---

### 🧠 ML Problem Framing

* **Type:** Binary Classification
* **Algorithm:** Logistic Regression
* **Target Variable:**

  * `Fully Paid → 0`
  * `Charged Off → 1`

---

### 📂 Dataset

* 📌 Source: LoanTap internal data
* 📊 Rows: ~396,000
* 📈 Features: 27 variables
* Includes:

  * Borrower profile
  * Loan details
  * Credit history

---

### 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

### 🔍 Exploratory Data Analysis (EDA)

#### ✔ Key Steps:

* Missing value analysis
* Outlier detection using boxplots
* Distribution analysis (histograms + KDE)
* Bivariate analysis with target variable
---
