

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.inspection import partial_dependence
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (TabPanel,ColorBar, BasicTicker, PrintfTickFormatter, HoverTool,InlineStyleSheet,GlobalInlineStyleSheet,
    FileInput, Select, MultiSelect, Button, ColumnDataSource, Div,CustomJS,
    Tabs
)
from bokeh.plotting import figure
from bokeh.palettes import Viridis256

import base64
import io
from bokeh.transform import linear_cmap

curdoc().theme = 'dark_minimal'

# Try to import xgboost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False



gstyle = GlobalInlineStyleSheet(css=""" html, body, .bk, .bk-root {background-color: #343838; margin: 0; padding: 0; height: 100%; color: white; font-family: 'Consolas', 'Courier New', monospace; } .bk { color: white; } .bk-input, .bk-btn, .bk-select, .bk-slider-title, .bk-headers, .bk-label, .bk-title, .bk-legend, .bk-axis-label { color: white !important; } .bk-input::placeholder { color: #aaaaaa !important; } """)
style2 = InlineStyleSheet(css=""" .bk-input { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1px solid #3c3c3c; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } /* Input Hover */ .bk-input:hover { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232;        /* Red border */ box-shadow: 0 0 9px 2px #ff3232cc;  /* Red glow */ border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } /* Input Focus */ .bk-input:focus { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232; box-shadow: 0 0 11px 3px #ff3232dd; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } .bk-input:active { outline: none; background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232; box-shadow: 0 0 14px 3px #ff3232; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } .bk-input:-webkit-autofill { background-color: #1e1e1e !important; color: #d4d4d4 !important; -webkit-box-shadow: 0 0 0px 1000px #1e1e1e inset !important; -webkit-text-fill-color: #d4d4d4 !important; } """)
button_style = InlineStyleSheet(css=""" .bk-btn { background-color: #46b4f9; color: #1e1e2e; font-weight: bold; border: 10px solid #46b4f9; border-radius: 6px; transform: rotate(0deg); box-shadow: none; transition: all 0.3s ease-in-out; } /* 🟦 Hover: #1e1e2e + rotate */ .bk-btn:hover { background-color: #1e1e2e; border-color: #1e1e2e; color: #46b4f9; transform: rotate(3deg); box-shadow: 0 0 15px 3px #46b4f9; } /* 🔴 Active (click hold): red + stronger rotate */ .bk-btn:active { background-color: red; border-color: red; transform: rotate(6deg); box-shadow: 0 0 15px 3px red; } """)

base_variables = """ :host { /* CSS Custom Properties for easy theming */ --primary-color: #8b5cf6; --secondary-color: #06b6d4; --background-color: #1f2937; --surface-color: #343838; --text-color: #f9fafb; --accent-color: #f59e0b; --danger-color: #ef4444; --success-color: #10b981; --border-color: #4b5563; --hover-color: #6366f1; background: none !important; } """
file_input_style = InlineStyleSheet(css=base_variables + """ :host input[type="file"] { background: var(--surface-color) !important; color: var(--text-color) !important; border: 2px dashed var(--border-color) !important; border-radius: 6px !important; padding: 20px !important; font-size: 14px !important; cursor: pointer !important; transition: all 0.2s ease !important; } :host input[type="file"]:hover { border-color: var(--primary-color) !important; background: rgba(139, 92, 246, 0.05) !important; } :host input[type="file"]::file-selector-button { background: var(--primary-color) !important; color: white !important; border: none !important; border-radius: 4px !important; padding: 8px 16px !important; margin-right: 12px !important; cursor: pointer !important; font-weight: 600 !important; } """)
fancy_div_style = InlineStyleSheet(css=""" :host { position: relative; background: #21233a; color: #fff; border-radius: 12px; padding: 18px 28px; text-align: center; overflow: hidden; box-shadow: 0 6px 10px rgba(197, 153, 10, 0.2); } :host::after { content: ''; position: absolute; top: 0; left: -80%; width: 200%; height: 100%; background: linear-gradient(120deg, transparent 40%, rgba(255, 252, 71, 0.416) 50%, transparent 60%); animation: shimmer 2.2s infinite; pointer-events: none; border-radius: inherit; } @keyframes shimmer { 0%   { left: -80%; } 100% { left: 100%; } } """)





# ---- Helper Functions ----
def parse_contents(contents, filename):
    try:
        decoded = base64.b64decode(contents)
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(decoded))
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(io.BytesIO(decoded))
        try:
            return pd.read_csv(io.BytesIO(decoded))
        except Exception:
            return pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")

def is_classification(y, threshold=10):
    if y.dtype.kind in 'biu' and y.nunique() <= threshold:
        return True
    if y.dtype.name == 'category':
        return True
    if y.dtype == object:
        return True
    return False

def get_partial_dependence_plot(model, X, feature, y, is_clf):
    if not np.issubdtype(X[feature].dtype, np.number):
        return None
    try:
        pd_result = partial_dependence(model, X, [feature], kind="average")
        xs = pd_result['values'][0]
        ys = pd_result['average'][0]
        p = figure(height=350, width=550, title=f'Partial Dependence: {feature}',
                   x_axis_label=feature, y_axis_label='Prediction',background_fill_color="#343838",border_fill_color="#343838",)
        p.line(xs, ys, line_width=3, line_color="orange")
        return p
    except Exception as e:
        print(f"Error creating partial dependence plot for {feature}: {e}")
        return None

# ---- Bokeh UI Components ----
title = Div(
    text="""
    <span style="display:block;color:deepskyblue;font-size:44px;font-weight:bold;">Phoebe</span>
    <span style="display:block;color:orange;font-size:18px;margin-top:-14px;">Relationship ML Explorer</span>
    """,
    styles={'width': '180px', 'background-color': 'black', 'padding': '10px', 'border-radius': '10px'},
    stylesheets=[fancy_div_style],
)
file_input = FileInput(accept=".csv,.xls,.xlsx",    stylesheets=[file_input_style])
columns_select = MultiSelect(title="Select Input Variables", options=[], size=8,stylesheets=[style2])
target_select = Select(title="Select Target Variable", options=[],stylesheets=[style2])
model_options = [
    ("Random Forest", "Random Forest"),
    ("Logistic Regression", "Logistic Regression"),
    ("Linear Regression", "Linear Regression"),
]
if XGB_AVAILABLE:
    model_options += [("XGBoost", "XGBoost")]

model_select = Select(title="Model", options=[k for k, _ in model_options], value="Random Forest",stylesheets=[style2])
analyze_btn = Button(label="Analyze", button_type="primary", disabled=True,stylesheets=[button_style])

validation_div = Div(text="", styles={'font-size': '14px'})
feature_importance_src = ColumnDataSource(data=dict(features=[], importances=[]))
feature_importance_fig = figure(
    x_range=[], height=450, title="Feature Importances",
    toolbar_location=None, tools="",background_fill_color="#343838",border_fill_color="#343838",
)
ee=feature_importance_fig.vbar(
    x='features', top='importances', width=0.8, source=feature_importance_src, color = "deepskyblue", hover_color="orange", hover_line_color="red",
)
feature_importance_fig.xaxis.major_label_orientation = 1
# Add hover
hover = HoverTool(
    tooltips=[
        ("features", "@features"), ("importances", "@importances")
    ],
    mode='mouse',
    renderers=[ee]
)
feature_importance_fig.add_tools(hover)

correlation_src = ColumnDataSource(data=dict(x=[], y=[], colors=[], text=[]))
correlation_fig = figure(
    height=450, width=650, x_range=[], y_range=[],
    title="Correlation Matrix", tools="",background_fill_color="#343838",border_fill_color="#343838",
)
correlation_fig.xaxis.major_label_orientation = 0.9 
partial_dep_div = Div(text="<b>Partial Dependence Plots</b>")
partial_dep_tabs = Tabs(tabs=[])

state = {'df': None, 'filename': None, 'analyzed': False}

def file_uploaded(attr, old, new):
    try:
        if not file_input.value:
            validation_div.text = "<span style='color:red'>No file selected</span>"
            return
        filename = file_input.filename
        if not filename:
            filename = "uploaded_file.csv"
        df = parse_contents(file_input.value, filename)
        df = df.loc[:, ~df.columns.duplicated()]
        state['df'] = df
        state['filename'] = filename
        columns = list(df.columns)
        target_select.options = columns
        columns_select.options = columns
        analyze_btn.disabled = False
        validation_div.text = (
            f"<b>✓ Loaded file:</b> {filename}<br>"
            f"<b>Rows:</b> {df.shape[0]} <b>Columns:</b> {df.shape[1]}<br>"
            f"<b>Columns:</b> {', '.join(columns[:5])}"
            f"{'...' if len(columns) > 5 else ''}"
        )
    except Exception as e:
        analyze_btn.disabled = True
        validation_div.text = f"<span style='color:red'>Error loading file: {str(e)}</span>"

def filename_updated(attr, old, new):
    if file_input.value and file_input.filename:
        file_uploaded(attr, old, new)

file_input.on_change('value', file_uploaded)
file_input.on_change('filename', filename_updated)

def analyze():
    try:
        df = state['df']
        target = target_select.value
        inputs = columns_select.value
        model_name = model_select.value

        if df is None:
            validation_div.text = "<span style='color:red'>Please upload a file first.</span>"
            return
        if not target:
            validation_div.text = "<span style='color:red'>Please select a target variable.</span>"
            return
        if not inputs:
            validation_div.text = "<span style='color:red'>Please select input variables.</span>"
            return

        use_cols = [target] + inputs
        df_use = df[use_cols].dropna()
        if df_use.empty:
            validation_div.text = "<span style='color:red'>No data remaining after removing missing values.</span>"
            return
        y = df_use[target]
        X = df_use[inputs]
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if X_numeric[col].dtype == object:
                numeric_col = pd.to_numeric(X_numeric[col], errors='coerce')
                if numeric_col.notna().sum() > len(numeric_col) * 0.5:
                    X_numeric[col] = numeric_col
                else:
                    X_numeric = X_numeric.drop(columns=[col])
        X_numeric = X_numeric.dropna(axis=1, how='all')
        if X_numeric.empty:
            validation_div.text = "<span style='color:red'>No valid numeric input variables found.</span>"
            return
        df_final = pd.concat([y, X_numeric], axis=1).dropna()
        if df_final.empty:
            validation_div.text = "<span style='color:red'>No data remaining after cleaning.</span>"
            return
        y_clean = df_final[target]
        X_clean = df_final[list(X_numeric.columns)]

        is_clf = is_classification(y_clean)
        # Model Selection
        if model_name == "Random Forest":
            if is_clf:
                y_enc = pd.factorize(y_clean)[0] if y_clean.dtype == object else y_clean.astype(int)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                score_type = "accuracy"
                score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='accuracy').mean()
            else:
                y_enc = pd.to_numeric(y_clean, errors='coerce')
                if y_enc.isna().any():
                    validation_div.text = "<span style='color:red'>Target variable contains non-numeric values.</span>"
                    return
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                score_type = "R²"
                score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='r2').mean()
        elif model_name == "Logistic Regression":
            y_enc = pd.factorize(y_clean)[0] if y_clean.dtype == object else y_clean.astype(int)
            model = LogisticRegression(max_iter=1000)
            score_type = "accuracy"
            score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='accuracy').mean()
        elif model_name == "Linear Regression":
            y_enc = pd.to_numeric(y_clean, errors='coerce')
            if y_enc.isna().any():
                validation_div.text = "<span style='color:red'>Target variable contains non-numeric values.</span>"
                return
            model = LinearRegression()
            score_type = "R²"
            score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='r2').mean()
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            if is_clf:
                y_enc = pd.factorize(y_clean)[0] if y_clean.dtype == object else y_clean.astype(int)
                model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
                score_type = "accuracy"
                score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='accuracy').mean()
            else:
                y_enc = pd.to_numeric(y_clean, errors='coerce')
                if y_enc.isna().any():
                    validation_div.text = "<span style='color:red'>Target variable contains non-numeric values.</span>"
                    return
                model = XGBRegressor(n_estimators=100, random_state=42)
                score_type = "R²"
                score = cross_val_score(model, X_clean, y_enc, cv=5, scoring='r2').mean()
        else:
            validation_div.text = f"<span style='color:red'>Selected model '{model_name}' not available.</span>"
            return

        model.fit(X_clean, y_enc)
        importances = getattr(model, "feature_importances_", np.ones(X_clean.shape[1]))
        sorted_idx = np.argsort(importances)[::-1]
        # Get top numeric features for partial dependence
        top_features = []
        for i in sorted_idx[:3]:
            feat_name = X_clean.columns[i]
            if np.issubdtype(X_clean[feat_name].dtype, np.number):
                top_features.append(feat_name)
        # Feature importance chart
        feat_names = [X_clean.columns[i] for i in sorted_idx]
        feat_imps = importances[sorted_idx]
        feature_importance_src.data = dict(features=feat_names, importances=feat_imps)
        feature_importance_fig.x_range.factors = feat_names

        # Validation score
        validation_div.text = (
            f"<b>✓ Analysis Complete!</b><br>"
            f"<b>Model:</b> {model_name} {'Classifier' if is_clf else 'Regressor'}<br>"
            f"<b>Cross-validated {score_type}:</b> {score:.3f}<br>"
            f"<b>Training samples:</b> {len(X_clean)}"
        )
        if len(X_clean.columns) > 1:
            update_correlation_heatmap(X_clean, y_clean)
        else:
            correlation_fig.x_range.factors = []
            correlation_fig.y_range.factors = []
            correlation_fig.renderers = []
            correlation_fig.right = []
        # Partial dependence tabs
        tabs = []
        for feat in top_features:
            p = get_partial_dependence_plot(model, X_clean, feat, y_enc, is_clf)
            if p:
                tabs.append(TabPanel(child=p, title=feat))
        partial_dep_tabs.tabs = tabs
        state['analyzed'] = True
    except Exception as e:
        validation_div.text = f"<span style='color:red'>Analysis error: {str(e)}</span>"
        print(f"Analysis error: {e}")

analyze_btn.on_click(analyze)



def update_correlation_heatmap(X_clean, y_clean):
    # Build correlation DataFrame
    corr_data = pd.concat([X_clean, y_clean], axis=1)
    corr = corr_data.corr().fillna(0)
    corr_cols = corr.columns.tolist()
    corr_vals = corr.values

    # Prepare data for plotting
    n = len(corr_cols)
    xname = []
    yname = []
    value = []

    for i in range(n):
        for j in range(n):
            xname.append(corr_cols[i])
            yname.append(corr_cols[j])
            value.append(corr_vals[j, i])  # rows: y, cols: x

    # Build ColumnDataSource
    new_correlation_src = dict(
        x=xname,
        y=yname,
        value=value,
    )
    correlation_src.data = new_correlation_src

    # Set factors for axes
    correlation_fig.x_range.factors = corr_cols
    correlation_fig.y_range.factors = list(reversed(corr_cols))

    # Remove any old renderers
    correlation_fig.renderers = []

    # Create a mapper for color
    mapper = linear_cmap(
        field_name='value',
        palette=Viridis256,
        low=-1,
        high=1
    )

    # Draw rectangles
    r = correlation_fig.rect(
        x='x', y='y',
        width=1, height=1,
        source=correlation_src,
        fill_color=mapper,
        line_color=None
    )

    # Add hover
    hover = HoverTool(
        tooltips=[
            ("X", "@x"), ("Y", "@y"), ("Correlation", "@value{0.2f}")
        ],
        mode='mouse',
        renderers=[r]
    )
    correlation_fig.add_tools(hover)

    # Add colorbar
    color_bar = ColorBar(
        color_mapper=mapper['transform'],
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=10),
        formatter=PrintfTickFormatter(format="%.2f"),
        label_standoff=8,
        title="Corr.",background_fill_color="#343838",
    )
    # Remove previous colorbars
    correlation_fig.right = []
    correlation_fig.add_layout(color_bar, 'right')




screenshot_btn = Button(label="Export Screenshot / PDF", button_type="success", stylesheets=[button_style])
screenshot_btn.js_on_click(CustomJS(code="window.print();"))


upload_panel = column(title, file_input, )
select_panel = row(column(model_select, target_select), columns_select, column(analyze_btn,screenshot_btn),validation_div)
output_panel = row(
    feature_importance_fig, 
    correlation_fig, 
    column(partial_dep_div, 
    partial_dep_tabs)
)

layout = column(row(upload_panel, select_panel), output_panel,stylesheets=[gstyle])

curdoc().add_root(layout )
curdoc().title = "Phoebe: Relationship ML Explorer"
