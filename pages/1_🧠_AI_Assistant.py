import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np
from dashboard.localization import t, all_translations

if 'lang' not in st.session_state:
    st.session_state.lang = 'ru'

st.set_page_config(page_title=t("page_assistant_title"), layout="wide", page_icon="🤖")

st.title(t("page_assistant_title"))
st.markdown(t("page_assistant_desc"))

@st.cache_data
def load_data():
    try:
        features_df = pd.read_csv("data/features_full.csv", parse_dates=["date"])
        model = joblib.load("models/lightgbm_demand_model.pkl")
        feature_order = joblib.load("models/feature_order.pkl")
        cat_levels = joblib.load("models/categorical_levels.pkl")
        return features_df, model, feature_order, cat_levels
    except FileNotFoundError as e:
        st.error(f"❌ Ошибка загрузки файла: {e}.")
        return None, None, None, None

df, model, feature_order, cat_levels = load_data()
if df is None: st.stop()

st.sidebar.header(t("sidebar_header_analysis"))

def update_language_assistant():
    lang_name = st.session_state.lang_selector_assistant
    lang_code = next(code for code, data in all_translations.items() if data["lang_name"] == lang_name)
    st.session_state.lang = lang_code

current_lang_name_assistant = all_translations[st.session_state.lang]['lang_name']
available_lang_names_assistant = [data['lang_name'] for data in all_translations.values()]
current_index_assistant = available_lang_names_assistant.index(current_lang_name_assistant)

st.sidebar.selectbox(
    label=t("language_selector"),
    options=available_lang_names_assistant,
    index=current_index_assistant,
    on_change=update_language_assistant,
    key='lang_selector_assistant'
)

product_id = st.sidebar.selectbox(t("select_product"), df["product_id"].unique(), key="sim_product")
store_id = st.sidebar.selectbox(t("select_store"), df["store_id"].unique(), key="sim_store")

df_filtered = df[(df["product_id"] == product_id) & (df["store_id"] == store_id)].copy()

if df_filtered.empty:
    st.warning(t("not_enough_data"))
    st.stop()

X = df_filtered[feature_order].copy()
for col in feature_order:
    if col in cat_levels:
        X[col] = X[col].astype('category')
        known_categories = cat_levels[col]
        X[col] = X[col].cat.set_categories(known_categories)
        if X[col].isnull().any():
            default_category = known_categories[0] if len(known_categories) > 0 else None
            if default_category: X[col].fillna(default_category, inplace=True)
    elif X[col].dtype == 'object' or pd.api.types.is_bool_dtype(X[col]):
        X[col] = X[col].astype(bool).astype(int)
    elif pd.api.types.is_numeric_dtype(X[col]):
        X[col].fillna(0, inplace=True)
df_filtered["prediction"] = model.predict(X)

num_points = len(df_filtered)
indices = np.linspace(0, num_points - 1, 5, dtype=int) if num_points >= 5 else np.arange(num_points)
if len(indices) > 0:
    highlight_points = df_filtered.iloc[indices]
    highlight_dates = highlight_points['date'].dt.strftime('%Y-%m-%d').tolist()

    explanations = {
        highlight_dates[0]: {"title_key": "analysis_exp_1_title", "text_key": "analysis_exp_1_text", "color": "#273c75"},
        highlight_dates[1]: {"title_key": "analysis_exp_2_title", "text_key": "analysis_exp_2_text", "color": "#4cd137"},
        highlight_dates[2]: {"title_key": "analysis_exp_3_title", "text_key": "analysis_exp_3_text", "color": "#e84118"},
        highlight_dates[3]: {"title_key": "analysis_exp_4_title", "text_key": "analysis_exp_4_text", "color": "#192a56"},
        highlight_dates[4]: {"title_key": "analysis_exp_5_title", "text_key": "analysis_exp_5_text", "color": "#487eb0"}
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["target_sales_next_7d"], mode="lines", name=t("chart_fact"), line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["prediction"], mode="lines", name=t("chart_forecast"), line=dict(color="deeppink", dash="dot")))
    fig.add_trace(go.Scatter(x=highlight_points["date"], y=highlight_points["prediction"], mode="markers", name=t("chart_key_points"), marker=dict(color="orange", size=15, symbol='star')))
    fig.update_layout(title=t("model_quality_chart_title"), xaxis_title=t("chart_date_axis"), yaxis_title=t("chart_sales_axis"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    selected_date = st.radio(
        t("select_point_for_analysis"),
        highlight_dates,
        horizontal=True,
        key="ai_date_selector"
    )

    if selected_date:
        explanation_data = explanations.get(selected_date)
        if explanation_data:
            st.subheader(t(explanation_data["title_key"]))
            st.markdown(f'<div style="border-left: 5px solid {explanation_data["color"]}; padding: 10px; background-color: #f0f2f6; color: #333;">{t(explanation_data["text_key"])}</div>', unsafe_allow_html=True)
else:
    st.info(t("no_data_to_display"))