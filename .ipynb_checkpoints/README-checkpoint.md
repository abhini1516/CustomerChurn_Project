# Customer Churn Platform with SHAP Explainability

An end-to-end machine learning system that predicts telecom customer attrition risk, analyzes global and local risk metrics using Explainable AI (XAI), and calculates financial exposure parameters—deployed as an interactive Streamlit dashboard.

**Live Application:** [customerchurnproject-j73yuxfklhyeockqnik3ep.streamlit.app](https://customerchurnproject-j73yuxfklhyeockqnik3ep.streamlit.app)
**Dataset Reference:** IBM Telco Customer Churn (Kaggle Evaluation Asset)

---

## 📸 Dashboard Demo & Interface Sneak Peek

### Interactive Enterprise Risk Interface
![Dashboard Home Overview](data/cc2.png)
*The interactive interface accepts dynamic sidebar inputs, updates individual risk percentages instantly, and calculates at-risk Customer Lifetime Value (CLTV).*

### Core Explanatory Visualizations (EDA & SHAP)
| Exploratory Data Analysis | Explainable AI (SHAP Global Drivers) |
| :---: | :---: |
| ![Churn Distribution by Contract & ISP](data/cc4.png) | ![SHAP Beeswarm Plot](data/cc5.png) |
| *EDA highlights structural risk domains.* | *SHAP isolates exact mathematical game-theoretic feature impacts.* |

---

## 🎯 Project Value Proposition
Most baseline portfolio projects stop at an aggregate accuracy metric. This platform bridges the gap between pure machine learning and data product design by answering the exact operational questions an enterprise cares about: *Why is this specific customer flagged for cancellation, what is the exact revenue at risk, and what strategic intervention is required?*

Using **SHAP (SHapley Additive exPlanations)**, the underlying mathematical inference is decomposed into explicit feature attributions, converting a traditional black-box model into an actionable customer retention tool.

---

## 📊 Key Findings from EDA

**Dataset Overview:** 7,043 customers across 33 raw columns. After sanitizing `Total Charges` data types, handling hidden empty spaces, and dropping 7 administrative/leaky columns, the data was isolated into an 80/20 split — 5,634 training rows and 1,409 testing rows — with a 26.54% baseline churn rate in the training set.

* **Contract Type Vulnerability:** Month-to-month subscribers exhibit a severe attrition rate of **42.75%**, compared to **11.08%** for one-year contracts and just **2.87%** for two-year agreements. Transitioning users off short-term structures slashes churn probability by nearly 14x.
* **Product Line Operational Anomaly:** Premium Fiber Optic subscribers churn at a disproportionate rate of **42.09%** (contrasting against DSL at **18.69%**, and no internet service at **7.25%**). In a live enterprise ecosystem, this signals critical infrastructure instability or aggressive competitor pricing within the high-speed tier.
* **Temporal Inversion:** The global SHAP beeswarm analysis confirms that `Tenure Months` is the single strongest inverse driver of risk, with high tenure (red) pushing the SHAP value sharply negative (lower churn risk) and low tenure (blue) pushing it positive. Operational vulnerability is heavily concentrated within the first 12 months of account generation.
* **Secondary SHAP Drivers:** `Internet Service_Fiber optic`, `Dependents_Yes`, and `Contract_Two year` show meaningful directional pull — fiber optic customers and electronic-check payers skew toward higher risk, while two-year contracts and dependents skew toward retention.

---

## ⚙️ Data Pipeline & Feature Engineering
To ensure enterprise-grade reliability, the codebase enforces strict engineering constraints:
1. **Data Isolation (Zero Leakage):** The dataset is partitioned into an 80/20 train/test split immediately following ingestion. Exploratory analysis, feature distribution profiling, and transformer fitting were executed exclusively on the training matrix (5,634 rows × 25 features).
2. **High-Cardinality Pruning:** Geographic text identifiers (`City`, `Zip Code`, `Lat Long`) were systematically pruned to protect the feature space from dimensional explosion. Latitudinal and longitudinal coordinate metrics were retained to model geographic variance without introducing the curse of dimensionality.
3. **Pipeline Serialization:** Categorical attributes (`Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`) are handled via `OneHotEncoder(drop='first')` to eliminate multicollinearity. Numerical columns (`Tenure Months`, `Monthly Charges`, `Total Charges`, `Latitude`, `Longitude`, `CLTV`) are standardized using `StandardScaler`. The final production-engineered matrix scales the 25-feature training input up to **33 dense input nodes**.

---

## 📈 Model Performance & Evaluation Matrix

| Classifier Architecture | Test Accuracy | Churn Recall (Class 1) | Churn Precision (Class 1) | Performance Status |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Weighted)** | **0.75** | **0.78** | **0.51** | **Selected Production Asset** |
| Random Forest Classifier | 0.80 | 0.51 | 0.67 | Rejected (Fails to catch 49% of churn) |

### 🧠 Selection Architecture Note
**Churn Recall was prioritized over raw accuracy.** In customer retention optimization, missing an at-risk user (a false negative) results in a permanent loss of Customer Lifetime Value (CLTV). Conversely, flagging a loyal customer incorrectly (a false positive) merely results in minor automated outreach or a retention offer. Class imbalance was addressed using Scikit-Learn's structural class-weight balancing directly within the optimization loss function — which is exactly why Logistic Regression catches 78% of churners despite Random Forest's higher raw accuracy.

---

## 🛠️ Explainable AI Architecture (SHAP)
The platform uses an interventional game-theoretic `LinearExplainer` to compute SHAP values across the held-out test partition. The global evaluation ranks features based on absolute mean impact, proving that `Tenure Months`, `Total Charges`, and `Monthly Charges` are the dominant continuous drivers, while `Internet Service_Fiber optic`, `Contract_Two year`, and `Dependents_Yes` are the strongest categorical drivers of model confidence.

---

## 🖥️ Live Dashboard Walkthrough
The Streamlit app takes a customer's profile (gender, senior citizen status, partner/dependents, phone/internet services, contract type, billing method, tenure, monthly charges, and CLTV) and produces a real-time **Attrition Risk Score** alongside a **Financial Impact Assessment**:

* **Stable Account** (e.g. 31.19% risk) → flagged green, no immediate revenue at risk.
* **High Risk Action Required** (e.g. 69.64% risk) → flagged red, with the full CLTV ($3,500.00) surfaced as revenue exposed to churn.
* A month-to-month, electronic-check customer with 41 months tenure and a $3,669 CLTV shows a 46.37% risk score — illustrating how contract type and payment method shift the score even with long tenure.

This turns a static prediction into an actionable, dollar-quantified retention signal a business team can act on immediately.

---

## 📁 System Repository Directory

```text
CustomerChurn_Project/
├── data/
│   ├── Telco_customer_churn.xlsx   # Sourced raw evaluation dataset
│   ├── churn_model.pkl             # Serialized pipeline artifacts (Preprocessor, Weights, Feature Maps)
│   ├── cc2.png                     # Dashboard home overview screenshot
│   ├── cc4.png                     # EDA churn distribution bar charts
│   └── cc5.png                     # SHAP global feature importance (beeswarm) plot
├── Main_Pipeline.ipynb             # Experimental workspace, EDA plots, and SHAP modeling
├── app.py                          # Streamlit production application file
├── requirements.txt                # System dependency configuration file
├── LICENSE                         # MIT License
└── README.md                       # Comprehensive repository documentation
```

---

## 📄 License
This project is licensed under the [MIT License](https://github.com/abhini1516/CustomerChurn_Project/blob/main/LICENSE).

---

## 🤝 Get Involved
Interested in contributing or improving this project? You can:

* Fork the repository and submit pull requests.
* Report issues or bugs.
* Suggest new features or enhancements.
* Help improve the documentation.

Let's build a better customer retention tool together!

---

## 📫 Contact & Connect

* GitHub: [abhini1516](https://github.com/abhini1516)
* LinkedIn: [abhini-s](https://www.linkedin.com/in/abhini-s-220345281/)
* Email: [abhiniprojects7@gmail.com](mailto:abhiniprojects7@gmail.com)