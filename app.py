import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_curve, roc_auc_score, recall_score, f1_score
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Alert Classifier — PR 3",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #21262d;
        border-left: 4px solid #f85149;
        padding: 2rem 2.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #f85149;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #8b949e;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }

    .section-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        color: #f85149;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .metric-box {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1rem 1.5rem;
        min-width: 140px;
        text-align: center;
    }

    .metric-box .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #58a6ff;
    }

    .metric-box .label {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    .stSidebar {
        background: #161b22 !important;
        border-right: 1px solid #21262d;
    }

    .stSidebar [data-testid="stSidebarContent"] {
        padding: 1.5rem 1rem;
    }

    .info-box {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.88rem;
        color: #8b949e;
        line-height: 1.6;
    }

    .theory-q {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #f0883e;
        margin-bottom: 0.4rem;
    }

    .stButton > button {
        background: #f85149 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        background: #da3633 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #21262d;
        border-radius: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
    }

    .stTabs [aria-selected="true"] {
        color: #f85149 !important;
        border-bottom-color: #f85149 !important;
    }

    .plot-container {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem;
    }

    hr {
        border-color: #21262d;
        margin: 1.5rem 0;
    }

    .stSuccess {
        background-color: #1f4a2e !important;
        border-color: #238636 !important;
        color: #3fb950 !important;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .badge-red { background: #3d1f1f; color: #f85149; border: 1px solid #6d2b2b; }
    .badge-blue { background: #1f2d3d; color: #58a6ff; border: 1px solid #2b4a6b; }
    .badge-green { background: #1f3d2b; color: #3fb950; border: 1px solid #2b6b3d; }
    .badge-orange { background: #3d2b1f; color: #f0883e; border: 1px solid #6b4a2b; }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔴 Risk Alert Classifier</h1>
    <p>PR-3 · Logistic Regression · Decision Tree · Random Forest · Imbalanced Learning</p>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR NAV ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:2px;
color:#f85149; text-transform:uppercase; margin-bottom:1rem;">Navigation</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "",
    [
        "📖 Part A — Theory",
        "📊 Part B — Data Loading",
        "🧹 Part C — Preprocessing & LR",
        "⚖️ Part D — Imbalanced Data",
        "🌳 Part E — Trees",
        "🔧 Part F — Hyperparameter Tuning",
        "📈 Part G — ROC / AUC",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:2px;
color:#f85149; text-transform:uppercase; margin-bottom:0.7rem;">Upload Dataset</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    "Risk_Alert_Classifier_Dataset.csv",
    type=["csv"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div class="info-box" style="margin-top:0.5rem;">
Upload the <code>Risk_Alert_Classifier_Dataset.csv</code> file to run all live models.
Without it, theory sections still work.
</div>
""", unsafe_allow_html=True)


# ─── DATA LOADING HELPER ────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare(file):
    df = pd.read_csv(file)

    # Encode categoricals
    le_gender = LabelEncoder()
    le_region = LabelEncoder()
    le_emp = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"].astype(str))
    df["region"] = le_region.fit_transform(df["region"].astype(str))
    df["employment_type"] = le_emp.fit_transform(df["employment_type"].astype(str))

    drop_cols = ["risk_status"]
    if "last_transaction_date" in df.columns:
        drop_cols.append("last_transaction_date")

    X = df.drop(drop_cols, axis=1)
    y = df["risk_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

    return df, X, y, X_train_imp, X_test_imp, y_train, y_test


def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    return fig, ax


# ─── PART A — THEORY ─────────────────────────────────────────────────────────
if section == "📖 Part A — Theory":
    st.markdown('<div class="section-label">Part A — Theory Questions</div>', unsafe_allow_html=True)

    qs = {
        "Q1 — Logistic Regression": """
**Logistic Regression** is a supervised learning algorithm used to predict the probability of a **categorical outcome**.
It applies the **sigmoid function** to map predicted values between 0 and 1.

It is suitable for classification because it estimates class probabilities and assigns observations to classes
based on a decision threshold (default = 0.5).

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```
""",
        "Q2 — Classification Metrics": """
Classification performance metrics evaluate how well a model predicts class labels.
Common metrics include **Accuracy, Precision, Recall, F1-Score, and AUC-ROC**.

**Accuracy alone is insufficient** because it can be misleading for **imbalanced datasets**.
For example, if 95% of cases belong to one class, a model predicting only that class achieves 95% accuracy
but fails to identify the minority class — making it practically useless.
""",
        "Q3 — Type-I & Type-II Errors": """
- **Type-I Error (False Positive):** Predicting a risk when **no actual risk exists**.
  → e.g., identifying a healthy person as high-risk.

- **Type-II Error (False Negative):** Failing to predict a risk when it **actually exists**.
  → e.g., classifying a high-risk patient as low-risk.

In medical/fraud contexts, **Type-II Error is more dangerous** because missing a real risk can be costly.
""",
        "Q4 — Precision, Recall, F1, TPR, FPR": """
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct |
| **Recall (TPR)** | TP / (TP + FN) | Of actual positives, how many we caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean balancing Precision & Recall |
| **FPR** | FP / (FP + TN) | Actual negatives wrongly called positive |
""",
        "Q5 — ROC Curve & AUC": """
**ROC Curve** plots **True Positive Rate (TPR)** against **False Positive Rate (FPR)**
at different threshold values.

**AUC (Area Under the Curve)** measures the overall ability of the classifier:
- **AUC = 1.0** → Perfect classifier
- **AUC = 0.5** → Random guessing (diagonal line)
- **AUC > 0.8** → Generally considered good

Higher AUC = better ability to distinguish between classes.
""",
        "Q6 — Imbalanced Data": """
**Imbalanced data** occurs when one class has significantly more samples than another.

Models trained on such data tend to **favor the majority class** and may ignore the minority class.
As a result, the model can achieve high accuracy while performing **poorly on the minority class**
— leading to biased and unreliable predictions.

**Solutions:**
- Under-sampling (reduce majority class)
- Over-sampling (duplicate minority class)
- SMOTE (Synthetic Minority Oversampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
"""
    }

    for q, ans in qs.items():
        with st.expander(q, expanded=False):
            st.markdown(ans)


# ─── PART B — DATA LOADING ────────────────────────────────────────────────────
elif section == "📊 Part B — Data Loading":
    st.markdown('<div class="section-label">Part B — Data Loading & Exploration</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
        st.markdown("""
**Expected columns include:**
`age`, `income`, `gender`, `region`, `employment_type`, `credit_score`, `loan_amount`,
`loan_tenure`, `num_dependents`, `risk_status` (target), etc.
        """)
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Target Classes", y.nunique())

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Class Distribution", "🔢 Stats"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)

        with tab2:
            vc = y.value_counts()
            fig, ax = dark_fig(figsize=(6, 4))
            colors = ['#f85149', '#58a6ff']
            bars = ax.bar(vc.index.astype(str), vc.values, color=colors, width=0.5, edgecolor='none')
            for bar, v in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{v:,}', ha='center', color='#e6edf3', fontsize=10,
                        fontfamily='monospace')
            ax.set_title("Class Distribution — risk_status")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=False)

            pct = y.value_counts(normalize=True) * 100
            st.markdown(f"**Class balance:** {pct.iloc[0]:.1f}% vs {pct.iloc[1]:.1f}%")

        with tab3:
            st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
        st.markdown("**Missing values per column:**")
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Column", "Missing"]
        null_df = null_df[null_df["Missing"] > 0]
        if null_df.empty:
            st.success("✅ No missing values found.")
        else:
            st.dataframe(null_df, use_container_width=True)


# ─── PART C — PREPROCESSING & LOGISTIC REGRESSION ───────────────────────────
elif section == "🧹 Part C — Preprocessing & LR":
    st.markdown('<div class="section-label">Part C — Preprocessing & Logistic Regression</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        st.markdown("### Q9 — Preprocessing Steps")
        with st.expander("View preprocessing details", expanded=True):
            st.markdown("""
**Steps performed:**
1. **Label Encoding** — `gender`, `region`, `employment_type` converted to numeric
2. **Drop date column** — `last_transaction_date` removed (if present)
3. **KNN Imputer** — Missing values filled using 5 nearest neighbours
4. **Train/Test Split** — 80% train, 20% test, stratified by target
            """)
            c1, c2 = st.columns(2)
            c1.metric("Training Samples", X_train_imp.shape[0])
            c2.metric("Testing Samples",  X_test_imp.shape[0])

        st.markdown("---")
        st.markdown("### Q10 & Q11 — Logistic Regression")

        if st.button("▶ Train Logistic Regression"):
            with st.spinner("Training..."):
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train_imp, y_train)
                y_pred = lr.predict(X_test_imp)
                acc = accuracy_score(y_test, y_pred)
                cm  = confusion_matrix(y_test, y_pred)
                TN, FP, FN, TP = cm.ravel()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy",  f"{acc:.4f}")
            col2.metric("Type-I (FP)", FP)
            col3.metric("Type-II (FN)", FN)
            col4.metric("True Positives", TP)

            st.markdown("**Confusion Matrix:**")
            fig, ax = dark_fig(figsize=(5, 4))
            im = ax.imshow(cm, cmap='RdYlBu')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred 0', 'Pred 1'], color='#e6edf3')
            ax.set_yticklabels(['Actual 0', 'Actual 1'], color='#e6edf3')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white', fontsize=16, fontfamily='monospace', fontweight='bold')
            ax.set_title("Confusion Matrix — Logistic Regression")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

            st.markdown("**Classification Report:**")
            cr = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(cr).transpose().round(4), use_container_width=True)


# ─── PART D — IMBALANCED DATA ────────────────────────────────────────────────
elif section == "⚖️ Part D — Imbalanced Data":
    st.markdown('<div class="section-label">Part D — Handling Imbalanced Data</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        st.markdown("**Class distribution in training set:**")
        st.dataframe(y_train.value_counts().reset_index().rename(
            columns={"index": "Class", "risk_status": "Count"}), use_container_width=True)

        st.markdown("---")
        st.markdown("### Q14 — Resampling Techniques")
        st.info("Click below to train all 5 models (Original + 4 resampling methods).")

        if st.button("▶ Run All Sampling Methods"):
            results = {}

            with st.spinner("Training all models..."):
                # Samplers
                samplers = {
                    "Original":        (X_train_imp, y_train),
                    "Under Sampling":  RandomUnderSampler(random_state=42).fit_resample(X_train_imp, y_train),
                    "Over Sampling":   RandomOverSampler(random_state=42).fit_resample(X_train_imp, y_train),
                    "SMOTE":           SMOTE(random_state=42).fit_resample(X_train_imp, y_train),
                    "ADASYN":          ADASYN(random_state=42).fit_resample(X_train_imp, y_train),
                }

                for name, (Xs, ys) in samplers.items():
                    m = LogisticRegression(max_iter=5000, random_state=42)
                    m.fit(Xs, ys)
                    yp = m.predict(X_test_imp)
                    yprob = m.predict_proba(X_test_imp)[:, 1]
                    results[name] = {
                        "Recall":   recall_score(y_test, yp),
                        "F1 Score": f1_score(y_test, yp),
                        "AUC ROC":  roc_auc_score(y_test, yprob),
                        "Sample Count": len(ys),
                        "pred": yp,
                        "prob": yprob,
                    }

            comp_df = pd.DataFrame({
                k: {m: v[m] for m in ["Recall", "F1 Score", "AUC ROC"]}
                for k, v in results.items()
            }).T.round(4)

            st.markdown("### Q15 — Comparison Table")
            st.dataframe(comp_df.sort_values("Recall", ascending=False), use_container_width=True)

            # Bar chart comparison
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.patch.set_facecolor('#161b22')
            palette = ['#f85149', '#58a6ff', '#3fb950', '#f0883e', '#bc8cff']

            for idx, metric in enumerate(["Recall", "F1 Score", "AUC ROC"]):
                ax = axes[idx]
                ax.set_facecolor('#0d1117')
                vals = comp_df[metric]
                bars = ax.bar(range(len(vals)), vals.values, color=palette,
                              edgecolor='none', width=0.6)
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(vals.index, rotation=30, ha='right',
                                   color='#8b949e', fontsize=8)
                ax.set_title(metric, color='#e6edf3')
                ax.tick_params(colors='#8b949e')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#21262d')
                for bar, v in zip(bars, vals.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{v:.3f}', ha='center', color='#e6edf3', fontsize=8,
                            fontfamily='monospace')

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            best = comp_df["Recall"].idxmax()
            st.success(f"✅ Best model by Recall: **{best}** ({comp_df.loc[best, 'Recall']:.4f})")


# ─── PART E — DECISION TREE & RANDOM FOREST ──────────────────────────────────
elif section == "🌳 Part E — Trees":
    st.markdown('<div class="section-label">Part E — Decision Tree & Random Forest</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        if st.button("▶ Train Both Models"):
            with st.spinner("Training Decision Tree..."):
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train_imp, y_train)
                y_pred_dt = dt.predict(X_test_imp)
                train_pred_dt = dt.predict(X_train_imp)
                dt_train_acc = accuracy_score(y_train, train_pred_dt)
                dt_test_acc  = accuracy_score(y_test,  y_pred_dt)

            with st.spinner("Training Random Forest..."):
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_imp, y_train)
                y_pred_rf = rf.predict(X_test_imp)
                rf_train_acc = accuracy_score(y_train, rf.predict(X_train_imp))
                rf_test_acc  = accuracy_score(y_test,  y_pred_rf)

            st.markdown("### Q17 & Q19 — Overfitting Analysis")
            comp = pd.DataFrame({
                "Model": ["Decision Tree", "Random Forest"],
                "Training Accuracy": [dt_train_acc, rf_train_acc],
                "Testing Accuracy":  [dt_test_acc,  rf_test_acc],
                "Gap (Train-Test)":  [dt_train_acc - dt_test_acc, rf_train_acc - rf_test_acc]
            }).round(4)
            st.dataframe(comp, use_container_width=True)

            if dt_train_acc - dt_test_acc > 0.05:
                st.warning("⚠️ Decision Tree shows signs of **overfitting** (large train-test gap).")
            else:
                st.success("✅ Decision Tree generalises well.")

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Decision Tree — Classification Report**")
                cr_dt = pd.DataFrame(
                    classification_report(y_test, y_pred_dt, output_dict=True)
                ).transpose().round(4)
                st.dataframe(cr_dt, use_container_width=True)

            with col2:
                st.markdown("**Random Forest — Classification Report**")
                cr_rf = pd.DataFrame(
                    classification_report(y_test, y_pred_rf, output_dict=True)
                ).transpose().round(4)
                st.dataframe(cr_rf, use_container_width=True)

            # Feature importance
            st.markdown("---")
            st.markdown("### Feature Importance — Random Forest")
            fi = pd.Series(rf.feature_importances_, index=X_train_imp.columns).sort_values(ascending=False).head(12)
            fig, ax = dark_fig(figsize=(10, 4))
            colors = ['#f85149' if i < 3 else '#58a6ff' for i in range(len(fi))]
            ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor='none')
            ax.set_xlabel("Importance")
            ax.set_title("Top-12 Feature Importances")
            st.pyplot(fig, use_container_width=True)


# ─── PART F — HYPERPARAMETER TUNING ──────────────────────────────────────────
elif section == "🔧 Part F — Hyperparameter Tuning":
    st.markdown('<div class="section-label">Part F — Hyperparameter Tuning</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        st.markdown("### Q20 — RandomizedSearchCV")

        col1, col2 = st.columns(2)
        with col1:
            dt_n_iter = st.slider("DT — n_iter", 5, 30, 15)
        with col2:
            rf_n_iter = st.slider("RF — n_iter", 5, 30, 15)

        if st.button("▶ Run Hyperparameter Search"):
            results_hp = {}

            # Baseline models
            with st.spinner("Training baseline Decision Tree..."):
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train_imp, y_train)
                dt_base_acc = accuracy_score(y_test, dt.predict(X_test_imp))

            with st.spinner("Training baseline Random Forest..."):
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_imp, y_train)
                rf_base_acc = accuracy_score(y_test, rf.predict(X_test_imp))

            # DT Randomized Search
            with st.spinner("DT RandomizedSearchCV..."):
                dt_params = {
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['gini', 'entropy']
                }
                dt_random = RandomizedSearchCV(
                    DecisionTreeClassifier(random_state=42),
                    dt_params, n_iter=dt_n_iter, cv=5,
                    scoring='accuracy', random_state=42, n_jobs=-1
                )
                dt_random.fit(X_train_imp, y_train)
                dt_tuned_acc = accuracy_score(y_test, dt_random.best_estimator_.predict(X_test_imp))
                results_hp["DT Best Params"] = dt_random.best_params_

            # RF Randomized Search
            with st.spinner("RF RandomizedSearchCV..."):
                rf_params = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
                rf_random = RandomizedSearchCV(
                    RandomForestClassifier(random_state=42),
                    rf_params, n_iter=rf_n_iter, cv=5,
                    scoring='accuracy', random_state=42, n_jobs=-1
                )
                rf_random.fit(X_train_imp, y_train)
                rf_tuned_acc = accuracy_score(y_test, rf_random.best_estimator_.predict(X_test_imp))
                results_hp["RF Best Params"] = rf_random.best_params_

            # Q21 GridSearchCV
            with st.spinner("RF GridSearchCV..."):
                grid_params = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    grid_params, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid.fit(X_train_imp, y_train)
                grid_acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test_imp))

            # Q22 Final comparison
            st.markdown("### Q22 — Final Model Comparison")
            final_comp = pd.DataFrame({
                "Model": [
                    "Decision Tree (Untuned)",
                    "Decision Tree (Tuned)",
                    "Random Forest (Untuned)",
                    "Random Forest (Tuned)",
                    "Grid Search RF"
                ],
                "Accuracy": [dt_base_acc, dt_tuned_acc, rf_base_acc, rf_tuned_acc, grid_acc]
            }).round(4)

            st.dataframe(final_comp.sort_values("Accuracy", ascending=False), use_container_width=True)

            fig, ax = dark_fig(figsize=(10, 4))
            colors = ['#8b949e', '#f0883e', '#8b949e', '#58a6ff', '#f85149']
            bars = ax.bar(final_comp["Model"], final_comp["Accuracy"],
                          color=colors, edgecolor='none', width=0.6)
            ax.set_ylim(min(final_comp["Accuracy"]) - 0.01, max(final_comp["Accuracy"]) + 0.01)
            ax.set_xticklabels(final_comp["Model"], rotation=20, ha='right', color='#8b949e', fontsize=9)
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Accuracy Comparison")
            for bar, v in zip(bars, final_comp["Accuracy"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                        f'{v:.4f}', ha='center', color='#e6edf3', fontsize=8,
                        fontfamily='monospace')
            st.pyplot(fig, use_container_width=True)

            with st.expander("Best Parameters Found"):
                for k, v in results_hp.items():
                    st.markdown(f"**{k}:**")
                    st.json(v)


# ─── PART G — ROC / AUC ──────────────────────────────────────────────────────
elif section == "📈 Part G — ROC / AUC":
    st.markdown('<div class="section-label">Part G — ROC Curves & AUC Comparison</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload the CSV file from the sidebar to proceed.")
    else:
        df, X, y, X_train_imp, X_test_imp, y_train, y_test = load_and_prepare(uploaded)

        if st.button("▶ Train All & Plot ROC Curves"):
            probs = {}
            preds = {}

            with st.spinner("Training all models for ROC curves..."):
                # LR Original
                lr = LogisticRegression(max_iter=5000, random_state=42)
                lr.fit(X_train_imp, y_train)
                probs["Logistic Regression"] = lr.predict_proba(X_test_imp)[:, 1]
                preds["Logistic Regression"] = lr.predict(X_test_imp)

                # Samplers
                for name, sampler in [
                    ("Under Sampling", RandomUnderSampler(random_state=42)),
                    ("Over Sampling",  RandomOverSampler(random_state=42)),
                    ("SMOTE",          SMOTE(random_state=42)),
                    ("ADASYN",         ADASYN(random_state=42)),
                ]:
                    Xs, ys = sampler.fit_resample(X_train_imp, y_train)
                    m = LogisticRegression(max_iter=5000, random_state=42)
                    m.fit(Xs, ys)
                    probs[name] = m.predict_proba(X_test_imp)[:, 1]
                    preds[name] = m.predict(X_test_imp)

                # RF
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_imp, y_train)
                probs["Random Forest"] = rf.predict_proba(X_test_imp)[:, 1]
                preds["Random Forest"] = rf.predict(X_test_imp)

                # GridSearchCV RF
                grid_params = {
                    'n_estimators': [100, 150], 'max_depth': [10, 15],
                    'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]
                }
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    grid_params, cv=3, scoring='accuracy', n_jobs=-1
                )
                grid.fit(X_train_imp, y_train)
                probs["Tuned RF (Grid)"] = grid.best_estimator_.predict_proba(X_test_imp)[:, 1]
                preds["Tuned RF (Grid)"] = grid.best_estimator_.predict(X_test_imp)

            # Q23 — ROC Plot
            st.markdown("### Q23 — ROC Curve Comparison")
            fig, ax = dark_fig(figsize=(10, 7))
            palette = ['#f85149', '#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#79c0ff', '#ffa657', '#e3b341']

            for (name, prob), color in zip(probs.items(), palette):
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc = roc_auc_score(y_test, prob)
                ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, linewidth=2)

            ax.plot([0, 1], [0, 1], 'k--', color='#30363d', linewidth=1, label="Random (AUC = 0.500)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve — All Models")
            ax.legend(loc='lower right', fontsize=8,
                      facecolor='#161b22', edgecolor='#21262d', labelcolor='#e6edf3')
            st.pyplot(fig, use_container_width=True)

            # Q24 — AUC Table
            st.markdown("### Q24 — AUC-ROC Table")
            auc_df = pd.DataFrame({
                "Model": list(probs.keys()),
                "AUC ROC": [roc_auc_score(y_test, p) for p in probs.values()]
            }).sort_values("AUC ROC", ascending=False).round(4)
            st.dataframe(auc_df, use_container_width=True)

            # Q25 — Recall Table
            st.markdown("### Q25 — Recall Comparison")
            recall_df = pd.DataFrame({
                "Model": list(preds.keys()),
                "Recall": [recall_score(y_test, p) for p in preds.values()]
            }).sort_values("Recall", ascending=False).round(4)
            st.dataframe(recall_df, use_container_width=True)

            best_auc = auc_df.iloc[0]
            best_recall = recall_df.iloc[0]
            st.success(
                f"🏆 Best AUC: **{best_auc['Model']}** ({best_auc['AUC ROC']:.4f}) | "
                f"Best Recall: **{best_recall['Model']}** ({best_recall['Recall']:.4f})"
            )


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
color:#30363d; padding:1rem 0;">
PR-3 · Risk Alert Classifier · Logistic Regression · Imbalanced Learning · Tree Models
</div>
""", unsafe_allow_html=True)