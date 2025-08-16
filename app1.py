import numpy as np
from typing import Union
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def auto_problem_type(y:pd.Series)-> str:
    if y.dtype=="object" or str(y.dtype).startswith("category"):
        return "classification"
    nunique=y.nunique(dropna=True)
    if nunique<=20:
        return "classification"
    return "regression"

def prepare_features(df:pd.DataFrame, target_col:str)->tuple[pd.DataFrame, Union[pd.Series, np.ndarray],str,dict]:
    X=df.drop(columns=[target_col])
    y=df[target_col]
    problem=auto_problem_type(y)
    
    cat_cols=X.select_dtypes(include=["object","category"]).columns.tolist()
    X_proc = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    encoders={}
    if problem=="classification":
        le=LabelEncoder()
        y_enc=le.fit_transform(y)
        encoders["label_encoder"]=le
    else:
        y_enc=y.astype(float).values
    return X_proc, y_enc, problem, encoders

def train_xgb(X: pd.DataFrame, y:Union[pd.Series, np.ndarray], problem:str, params:dict):
    if problem=="classification":
        model=xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth",4),
            learning_rate=params.get("learning_rate",0.1),
            colsample_bytree=params.get("subsample",1.0),
            random_state=params.get("random_state",42),
            n_jobs=-1,
            tree_method=params.get("tree_method","auto"),
            reg_lambda=params.get("reg_lambda",1.0),
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.07),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
            tree_method=params.get("tree_method", "auto"),
            reg_lambda=params.get("reg_lambda", 1.0),
        )

    model.fit(X,y)
    booster=model.get_booster()
    booster.feature_names=list(X.columns)
    booster.feature_types=None
    return model


def eval_model(model, X_test, y_test, problem:str):
    y_pred=model.predict(X_test)
    if problem=="classification":
        acc=accuracy_score(y_test,y_pred)
        f1=f1_score(y_test, y_pred, average="weighted")
        return {"accuracy": acc, "f1_weighted": f1}
    else:
        rmse=mean_squared_error(y_test, y_pred, squared=False)
        r2=r2_score(y_test, y_pred)
        return {"rmse":rmse, "r2": r2}

def get_num_trees(model) -> int:
    booster = model.get_booster()
    return len(booster.get_dump())



st.title("XGBoost Decision Tree Visualizer")
st.caption("Pick a dataset, train a model, explore individual trees.")
with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Choose data    source:", ["Demo — Iris (classification)", "Upload CSV"], index=0)
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV (<= 50MB)", type=["csv"])
        na_vals = st.text_input("Treat these as NA (comma-separated)", value="")
        na_list = [s.strip() for s in na_vals.split(",") if s.strip()]
    else:
        uploaded = None
        na_list = []
    st.divider()
    st.header("Train / Test Split")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
    st.divider()
    st.header("XGBoost Params")
    n_estimators = st.slider("n_estimators", 10, 500, 120, 10)
    max_depth = st.slider("max_depth", 1, 12, 4, 1)
    learning_rate = st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3], value=0.1)
    subsample = st.select_slider("subsample", options=[0.5, 0.7, 0.8, 0.9, 1.0], value=1.0)
    colsample_bytree = st.select_slider("colsample_bytree", options=[0.5, 0.7, 0.8, 0.9, 1.0], value=1.0)
    tree_method = st.selectbox("tree_method", ["auto", "hist", "exact"], index=0)
    st.divider()
    st.header("Graph Options")
    rankdir = st.selectbox("Tree orientation", ["LR", "TB"], index=0, help="LR = left→right, TB = top→bottom")
    yes_color = st.color_picker("Yes-branch color", value="#1f77b4")
    no_color = st.color_picker("No-branch color", value="#d62728")

if data_mode == "Demo — Iris (classification)":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "species"}, inplace=True)
    target_col = "species"
else:
    if uploaded is None:
        st.info("⬆️ Upload a CSV to continue, or switch to the demo dataset in the sidebar.")
        st.stop()
    try:
        df = pd.read_csv(uploaded, na_values=na_list)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
    st.write("**Preview:**", df.head())
    target_col = st.selectbox("Select target column", options=df.columns)
with st.expander("Data summary", expanded=False):
    st.write(df.describe(include="all").T)

X, y, problem, encoders = prepare_features(df, target_col)

# Allow user to override problem type
problem = st.radio("Problem type", [problem, "classification", "regression"], index=0, horizontal=True, help="Auto-detected type can be overridden.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y if problem == "classification" else None
)

params = dict(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=random_state,
    tree_method=tree_method,
)

with st.status("Training XGBoost model…", expanded=False):
    model = train_xgb(X_train, y_train, problem, params)

metrics = eval_model(model, X_test, y_test, problem)

# Metrics display
metric_cols = st.columns(len(metrics))
for i, (k, v) in enumerate(metrics.items()):
    if isinstance(v, float):
        metric_cols[i].metric(k.upper(), f"{v:.4f}")
    else:
        metric_cols[i].metric(k.upper(), str(v))
num_trees = get_num_trees(model)
st.subheader("Explore Trees")
if num_trees == 0:
    st.warning("Model has no trees to display.")
    st.stop()

left, right = st.columns([1, 3])
with left:
    tree_idx = st.number_input("Tree index", min_value=0, max_value=max(0, num_trees - 1), value=0, step=1)
    st.caption(f"This model has {num_trees} trees. For multiclass, XGBoost adds one tree per class per boosting round.")

with right:
    st.caption("Scroll/zoom the graph area as needed.")

booster = model.get_booster()
booster.feature_names = list(X.columns)
booster.feature_types = None

try:
    dot = xgb.to_graphviz(
        booster,
        num_trees=int(tree_idx),
        rankdir=rankdir,
        yes_color=yes_color,
        no_color=no_color,
        condition_node_params={"shape": "box", "style": "rounded,filled", "fillcolor": "#eef6ff"},
        leaf_node_params={"shape": "box", "style": "rounded,filled", "fillcolor": "#f3fff1"},
    )
    st.graphviz_chart(dot.source, use_container_width=True)
except Exception as e:
    st.error(f"Failed to render tree {tree_idx}: {e}")
with st.expander("Export tree"):
    dot_text = dot.source if 'dot' in locals() else ""
    st.download_button(
        label="Download DOT (Graphviz)",
        data=dot_text,
        file_name=f"xgb_tree_{int(tree_idx)}.dot",
        mime="text/vnd.graphviz",
        disabled=(dot_text == ""),
    )
    st.caption("You can render DOT to PNG/SVG with Graphviz locally if you want static images.")

