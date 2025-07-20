import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error
from localization import t, all_translations

# Инициализация языка в самом начале
if 'lang' not in st.session_state:
    st.session_state.lang = 'ru'

# --- Конфигурация страницы ---
st.set_page_config(
    page_title=t("page_title"),
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Стили ---
st.markdown("""
<style>
    .main { background-color: #F0F2F6; }
    .kpi-card {
        background-color: #FFFFFF; padding: 20px; border-radius: 10px;
        border-left: 7px solid #6a11cb; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center; height: 150px; display: flex; flex-direction: column; justify-content: center;
    }
    .kpi-card h3 { margin: 0; font-size: 1.1rem; color: #555; }
    .kpi-card .value { font-size: 2.2rem; font-weight: bold; color: #6a11cb; }
    .kpi-card .subtext { font-size: 0.9rem; color: #888; }
</style>
""", unsafe_allow_html=True)


# --- Загрузка данных (с кэшированием) ---
@st.cache_data
def load_data():
    try:
        sales = pd.read_csv("data/sales.csv", parse_dates=["date"])
        products = pd.read_csv("data/products.csv")
        stores = pd.read_csv("data/stores.csv")
        features = pd.read_csv("data/features_full.csv", parse_dates=["date"])
        model = joblib.load("models/lightgbm_demand_model.pkl")
        feature_order = joblib.load("models/feature_order.pkl")
        cat_levels = joblib.load("models/categorical_levels.pkl")
        total_records = len(sales)
        return sales, products, stores, features, model, feature_order, cat_levels, total_records
    except FileNotFoundError as e:
        st.error(f"❌ Ошибка загрузки файла: {e}.")
        return [None]*8

sales, products, stores, features, model, feature_order, cat_levels, total_records = load_data()
if sales is None: st.stop()


# --- Боковая панель ---
st.sidebar.title(t("sidebar_title"))

def update_language():
    lang_name = st.session_state.lang_selector
    lang_code = next(code for code, data in all_translations.items() if data["lang_name"] == lang_name)
    st.session_state.lang = lang_code

current_lang_name = all_translations[st.session_state.lang]['lang_name']
available_lang_names = [data['lang_name'] for data in all_translations.values()]
current_index = available_lang_names.index(current_lang_name)

st.sidebar.selectbox(
    label=t("language_selector"),
    options=available_lang_names,
    index=current_index,
    on_change=update_language,
    key='lang_selector'
)

st.sidebar.markdown("---")
st.sidebar.header(t("sidebar_header_analysis"))
store_id = st.sidebar.selectbox(t("select_store"), stores["store_id"].unique())
product_id = st.sidebar.selectbox(t("select_product"), products["product_id"].unique())
st.sidebar.markdown("---")
st.sidebar.header(t("sidebar_header_forecast"))
max_date_in_data = features['date'].max()
sel_date = st.sidebar.date_input(
    t("date_input_label"), value=max_date_in_data + timedelta(days=1),
    min_value=max_date_in_data + timedelta(days=1), max_value=max_date_in_data + timedelta(days=30)
)
predict_btn = st.sidebar.button(t("predict_button"))


# --- Основной интерфейс ---
st.title(t("main_title"))
st.markdown(t("main_subtitle", store=store_id, product=product_id))

start_time = time.time()
df_filtered_full = sales[(sales['store_id'] == store_id) & (sales['product_id'] == product_id)]
df_model_analysis = features[(features['store_id'] == store_id) & (features['product_id'] == product_id)].copy()
accuracy = 0
mae = 0
rmse = 0
if not df_model_analysis.empty:
    X_test = df_model_analysis[feature_order].copy()
    for col in feature_order:
        if col in cat_levels:
            X_test[col] = X_test[col].astype('category')
            known_categories = cat_levels[col]
            X_test[col] = X_test[col].cat.set_categories(known_categories)
            if X_test[col].isnull().any():
                default_category = known_categories[0] if len(known_categories) > 0 else None
                if default_category: X_test[col].fillna(default_category, inplace=True)
        elif X_test[col].dtype == 'object' or pd.api.types.is_bool_dtype(X_test[col]):
            X_test[col] = X_test[col].astype(bool).astype(int)
        elif pd.api.types.is_numeric_dtype(X_test[col]):
            X_test[col].fillna(0, inplace=True)
    df_model_analysis['prediction'] = model.predict(X_test)
    df_model_analysis['prediction'] = np.maximum(0, df_model_analysis['prediction'])
    actual = df_model_analysis['target_sales_next_7d'].dropna()
    if not actual.empty:
        predicted = df_model_analysis.loc[actual.index, 'prediction']
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        mask = actual != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(actual[mask], predicted[mask])
            accuracy = 100 * (1 - mape)
        else:
            accuracy = 100.0 if np.all(predicted == 0) else 0.0
num_records = len(df_filtered_full)
processing_time = time.time() - start_time

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.markdown(f'''<div class="kpi-card"><h3>{t("kpi_accuracy_title")}</h3><p class="value">{accuracy:.1f}%</p><p class="subtext">{t("kpi_accuracy_subtitle")}</p></div>''', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'''<div class="kpi-card"><h3>{t("kpi_speed_title")}</h3><p class="value">{processing_time:.3f} {t("seconds")}</p><p class="subtext">{t("kpi_speed_subtitle")}</p></div>''', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'''<div class="kpi-card"><h3>{t("kpi_records_title")}</h3><p class="value">{total_records/1000:.1f}K</p><p class="subtext">{t("kpi_records_subtitle")}</p></div>''', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([t("tab_sales_analysis"), t("tab_forecast"), t("tab_model_quality")])

with tab1:
    st.subheader(t("dynamics_title"))
    dynamics_period = st.selectbox(
        t("dynamics_period_selector"),
        options=[30, 60, 90, 120, 180],
        format_func=lambda x: t("last_days", days=x),
        index=2
    )
    if not df_filtered_full.empty:
        end_date_dyn = df_filtered_full['date'].max()
        start_date_dyn = end_date_dyn - pd.DateOffset(days=dynamics_period-1)
        df_filtered = df_filtered_full[(df_filtered_full['date'] >= start_date_dyn) & (df_filtered_full['date'] <= end_date_dyn)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['sales'], mode='lines', name=t("chart_sales"), line=dict(color='#6a11cb', width=2)))
        fig.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['sales'], name=t("chart_volume"), marker_color='#2575fc', opacity=0.3))
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy['rolling_mean'] = df_filtered_copy['sales'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(x=df_filtered_copy['date'], y=df_filtered_copy['rolling_mean'], mode='lines', name=t("chart_rolling_mean"), line=dict(color='orange', width=2, dash='dash')))
        fig.update_layout(title=t("sales_dynamics_chart_title", days=dynamics_period), xaxis_title=t("chart_date_axis"), yaxis_title=t("chart_sales_axis"), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader(t("deep_analysis_title"))
        df_analysis = df_filtered_full.copy()
        df_analysis['day_of_week'] = df_analysis['date'].dt.day_name()
        df_analysis['month'] = df_analysis['date'].dt.month_name()
        col1, col2 = st.columns(2)
        with col1:
            sales_by_day = df_analysis.groupby('day_of_week')['sales'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig_day = px.bar(sales_by_day, x=sales_by_day.index, y='sales', title=t("dow_sales_chart_title"), labels={'sales': t("dow_sales_chart_y_axis"), 'index': t("dow_sales_chart_x_axis")})
            fig_day.update_layout(showlegend=False)
            st.plotly_chart(fig_day, use_container_width=True)
        with col2:
            sales_by_month = df_analysis.groupby('month')['sales'].mean().reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']).dropna()
            fig_month = px.bar(sales_by_month, x=sales_by_month.index, y='sales', title=t("month_sales_chart_title"), labels={'sales': t("dow_sales_chart_y_axis"), 'index': t("month_sales_chart_x_axis")})
            fig_month.update_layout(showlegend=False)
            st.plotly_chart(fig_month, use_container_width=True)

        st.markdown("---")
        st.subheader(t("heatmap_title"))
        st.info(t("heatmap_info"))
        heatmap_col1, heatmap_col2 = st.columns(2)
        with heatmap_col1:
            hm_store_id = st.selectbox(t("heatmap_store_selector"), stores["store_id"].unique(), key="hm_store")
        with heatmap_col2:
            hm_period = st.selectbox(t("heatmap_period_selector"), [7, 14, 30, 60], format_func=lambda x: t("last_days", days=x), key="hm_period", index=2)
        df_store_sales = sales[sales['store_id'] == hm_store_id].merge(products, on='product_id')
        end_date_hm = df_store_sales['date'].max()
        start_date_hm = end_date_hm - pd.DateOffset(days=hm_period-1)
        df_heatmap_filtered = df_store_sales[(df_store_sales['date'] >= start_date_hm) & (df_store_sales['date'] <= end_date_hm)]
        if not df_heatmap_filtered.empty:
            df_heatmap_filtered = df_heatmap_filtered.copy()
            df_heatmap_filtered['day_of_week'] = df_heatmap_filtered['date'].dt.day_name()
            heatmap_data = df_heatmap_filtered.groupby(['category_id', 'day_of_week'])['sales'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot_table(values='sales', index='category_id', columns='day_of_week', aggfunc='sum')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)
            fig_heatmap = px.imshow(
                heatmap_pivot,
                labels=dict(x=t("heatmap_chart_x_axis"), y=t("heatmap_chart_y_axis"), color=t("heatmap_chart_color_axis")),
                title=t("heatmap_chart_title", store=hm_store_id, days=hm_period),
                color_continuous_scale=px.colors.sequential.Plasma, text_auto=True, aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning(t("no_data_for_heatmap"))
    else:
        st.info(t("no_data_to_display"))

with tab2:
    st.subheader(t("forecast_on_date_title", date=sel_date.strftime('%d.%m.%Y')))
    if predict_btn:
        with st.spinner(t("api_request_spinner")):
            api_url = "https://aurorix-api.onrender.com/predict"
            payload = {"store_id": store_id, "product_id": product_id, "date": sel_date.strftime("%Y-%m-%d")}
            try:
                # <<< ИЗМЕНЕНО: Таймаут увеличен до 60 секунд >>>
                response = requests.post(api_url, json=payload, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    forecast_value = result.get("forecast_sales_next_7d")
                    st.success(t("api_success"))
                    st.metric(label=t("forecast_metric_label"), value=f"{forecast_value} {t('pcs')}")
                    st.json(result)
                else:
                    st.error(t("api_error", status_code=response.status_code))
                    st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.error(t("api_connection_error", e=e))

with tab3:
    st.subheader(t("model_quality_title"))
    if not df_model_analysis.empty and 'prediction' in df_model_analysis.columns:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(t("mae_metric"), f"{mae:.2f}")
        m_col2.metric(t("rmse_metric"), f"{rmse:.2f}")
        m_col3.metric(t("accuracy_metric"), f"{accuracy:.2f}%", help=t("accuracy_help_text"))
        fig_quality = go.Figure()
        fig_quality.add_trace(go.Scatter(x=df_model_analysis['date'], y=df_model_analysis['target_sales_next_7d'], mode='lines', name=t("chart_fact"), line=dict(color='blue')))
        fig_quality.add_trace(go.Scatter(x=df_model_analysis['date'], y=df_model_analysis['prediction'], mode='lines', name=t("chart_forecast"), line=dict(color='deeppink', dash='dash')))
        fig_quality.update_layout(title=t("model_quality_chart_title"), xaxis_title=t("chart_date_axis"), yaxis_title=t("chart_sales_axis"))
        st.plotly_chart(fig_quality, use_container_width=True)
    else:
        st.warning(t("not_enough_data"))
