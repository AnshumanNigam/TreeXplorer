# Decision Tree Visualizer

### <ins>Problem Statement</ins>:- Understanding how decision tree–based models like XGBoost make predictions can often feel like a “black box.” While XGBoost is one of the most powerful machine learning algorithms, its inner workings are not easily interpretable for beginners or even practitioners who want deeper insights. Existing tools for tree visualization are either too complex, lack interactivity, or require heavy setup. There is a need for a tool that can:

Provide an intuitive visualization of XGBoost decision trees.

Help users understand feature splits, thresholds, and decision paths in a user-friendly way.

Allow interactive exploration of trained trees without diving into raw code or command-line tools.

Be lightweight, accessible, and usable directly in a browser.

Introducing XGBoost Decision Tree Visualizer 

Our Streamlit-based web app allows users to:

Train an XGBoost model on example datasets or custom CSV uploads.

Visualize individual decision trees using Graphviz integration.

Explore feature importance and split thresholds interactively.

Gain clear, educational insights into how gradient boosting works.

Key Features 

Upload your own dataset or use built-in samples.

Select specific trees from the XGBoost ensemble to visualize.

Display feature importance alongside the tree view.

Lightweight, browser-based Streamlit interface.

Easy to extend for educational, research, or project use.

Tech Stack 

Python (Scikit-learn, XGBoost, Pandas, NumPy)

Streamlit (interactive UI)

Graphviz (tree visualization)

Future Enhancements 

Interactive tree traversal (decision path) for user-given inputs.

Export tree plots and feature importance as PDF/PNG reports.

Add support for Random Forest and other tree-based models.
