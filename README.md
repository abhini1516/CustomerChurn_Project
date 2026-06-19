# Customer Churn Platform with SHAP Explainability

An end-to-end machine learning system that predicts telecom customer attrition risk, analyzes global and local risk metrics using Explainable AI (XAI), and calculates financial exposure parameters—deployed as an interactive Streamlit dashboard.

**Live Application:** [Insert your Streamlit Cloud Deployment URL Here]  
**Dataset Reference:** IBM Telco Customer Churn (Kaggle Evaluation Asset)

---

## 📸 Dashboard Demo & Interface Sneak Peek

### Interactive Enterprise Risk Interface
![Dashboard Home Dashboard Overview]([Add a screenshot path or image URL of your running Streamlit App here])
*The interactive interface accepts dynamic sidebar inputs, updates individual risk percentages instantly, and calculates at-risk Customer Lifetime Value (CLTV).*

### Core Explanatory Visualizations (EDA & SHAP)
| Exploratory Data Analysis | Explainable AI (SHAP Global Drivers) |
| :---: | :---: |
| ![Contract & ISP Analysis]([Add a screenshot path of your Step 2 Matplotlib/Seaborn plots here]) | ![SHAP Beeswarm Plot]([Add a screenshot path of your SHAP beeswarm chart here]) |
| *EDA highlights structural risk domains.* | *SHAP isolates exact mathematical game-theoretic feature impacts.* |

---

## 🎯 Project Value Proposition
Most baseline portfolio projects stop at an aggregate accuracy metric. This platform bridges the gap between pure machine learning and data product design by answering the exact operational questions an enterprise cares about: *Why is this specific customer flagged for cancellation, what is the exact revenue at risk, and what strategic intervention is required?*

Using **SHAP (SHapley Additive exPlanations)**, the underlying mathematical inference is decomposed into explicit feature attributions, converting a traditional black-box model into an actionable customer retention tool.

---

## 📊 Key Findings from EDA
* **Contract Type Vulnerability:** Month-to-month subscribers exhibit a severe attrition rate of **42.75%**, compared to a baseline of just **2.87%** for clients bound to two-year agreements. Transitioning users off short-term structures slashes churn probability by nearly 14x.
* **Product Line Operational Anomaly:** Premium Fiber Optic subscribers churn at a disproportionate rate of **42.09%** (contrasting against DSL at **18.69%**). In a live enterprise ecosystem, this signals critical infrastructure instability or aggressive competitor pricing within the high-speed tier.
* **Temporal Inversion:** The global SHAP beeswarm analysis confirms that `Tenure Months` is the single strongest inverse driver of risk. Operational vulnerability is heavily concentrated within the first 12 months of account generation.

---

## ⚙️ Data Pipeline & Feature Engineering
To ensure enterprise-grade reliability, the codebase enforces strict engineering constraints:
1. **Data Isolation (Zero Leakage):** The dataset is partitioned into an 80/20 train/test split immediately following ingestion. Exploratory analysis, feature distribution profiling, and transformer fitting were executed exclusively on the training matrix.
2. **High-Cardinality Pruning:** Geographic text identifiers (`City`, `Zip Code`, `Lat Long`) were systematically pruned to protect the feature space from dimensional explosion. Latitudinal and longitudinal coordinate metrics were retained to model geographic variance without introducing the curse of dimensionality.
3. **Pipeline Serialization:** Categorical attributes are handled cleanly via `OneHotEncoder(drop='first')` to eliminate multicollinearity. Continuous features are standardized using `StandardScaler`. The resulting structural matrix compresses a potential 7,500+ dimension space down to **33 dense input nodes**.

---

## 📈 Model Performance & Evaluation Matrix

| Classifier Architecture | Test Accuracy | Churn Recall (Class 1) | Churn Precision (Class 1) | Performance Status |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Weighted)** | **0.75** | **0.78** | **0.51** | **Selected Production Asset** |
| Random Forest Classifier | 0.80 | 0.51 | 0.67 | Rejected (Fails to catch 49% of churn) |

### 🧠 Selection Architecture Note
**Churn Recall was prioritized over raw accuracy.** In customer retention optimization, missing an at-risk user (a false negative) results in a permanent loss of Customer Lifetime Value (CLTV). Conversely, flagging a loyal customer incorrectly (a false positive) merely results in minor automated outreach or a retention offer. Class imbalance was addressed using Scikit-Learn's structural class-weight balancing directly within the optimization loss function.

---

## 🛠️ Explainable AI Architecture (SHAP)
The platform uses an interventional game-theoretic `LinearExplainer` to compute SHAP values across the held-out test partition. The global evaluation ranks features based on absolute mean impact, proving that text contract constraints, service tiers, and temporal tenure are the baseline mathematical drivers of model confidence. 

---

## 📁 System Repository Directory

```text
CustomerChurn_Project/
├── data/
│   ├── Telco_customer_churn.xlsx   # Sourced raw evaluation dataset
│   └── churn_model.pkl             # Serialized pipeline artifacts (Preprocessor, Weights, Feature Maps)
├── notebooks/
│   └── Main_Pipeline.ipynb         # Experimental workspace, EDA plots, and SHAP modeling
├── app.py                          # Streamlit production application file
├── requirements.txt                # System dependency configuration file
└── README.md                       # Comprehensive repository documentation