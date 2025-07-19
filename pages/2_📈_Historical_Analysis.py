import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import timedelta
from localization import t, all_translations  # <<< ИСПРАВЛЕНО

if 'lang' not in st.session_state:
    st.session_state.lang = 'ru'

st.set_page_config(page_title=t("page_history_title"), layout="wide", page_icon="📈")

st.title(t("page_history_title"))


@st.cache_data
def load_full_data():
    try:
        sales = pd.read_csv("data/sales.csv", parse_dates=["date"])
        products = pd.read_csv("data/products.csv")
        stores = pd.read_csv("data/stores.csv")
        full_data = sales.merge(products, on='product_id').merge(stores, on='store_id')
        return full_data
    except FileNotFoundError as e:
        st.error(f"❌ Ошибка загрузки файла: {e}.")
        return None


df = load_full_data()
if df is None: st.stop()

st.sidebar.header(t("sidebar_header_analysis"))


def update_language_history():
    lang_name = st.session_state.lang_selector_history
    lang_code = next(code for code, data in all_translations.items() if data["lang_name"] == lang_name)
    st.session_state.lang = lang_code


current_lang_name_history = all_translations[st.session_state.lang]['lang_name']
available_lang_names_history = [data['lang_name'] for data in all_translations.values()]
current_index_history = available_lang_names_history.index(current_lang_name_history)

st.sidebar.selectbox(
    label=t("language_selector"),
    options=available_lang_names_history,
    index=current_index_history,
    on_change=update_language_history,
    key='lang_selector_history'
)

all_stores_option = t("store_filter_all")
store_list = [all_stores_option] + sorted(df['store_id'].unique().tolist())
selected_store = st.sidebar.selectbox(t("select_store"), options=store_list)
period_days = st.sidebar.radio(
    t("dynamics_period_selector"), options=[7, 30, 90, 365],
    format_func=lambda x: t("last_days", days=x), horizontal=True, index=1
)

end_date = df['date'].max()
start_date = end_date - timedelta(days=period_days)
df_period = df[df['date'] >= start_date]
st.info(t("period_info", start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d')))

tab1, tab2 = st.tabs([t("tab_top_sales"), t("tab_store_comparison")])

with tab1:
    if selected_store == all_stores_option:
        st.subheader(t("top_sales_all_stores_title"))
        data_to_analyze = df_period
    else:
        st.subheader(t("top_sales_one_store_title", store=selected_store))
        data_to_analyze = df_period[df_period['store_id'] == selected_store]

    if not data_to_analyze.empty:
        top_10_products_ids = data_to_analyze.groupby('product_id')['sales'].sum().nlargest(10).index
        df_top10 = data_to_analyze[data_to_analyze['product_id'].isin(top_10_products_ids)].copy()

        top10_pivot = df_top10.pivot_table(
            index=['product_id', 'brand'],
            columns=df_top10['date'].dt.strftime('%Y-%m-%d'),
            values='sales'
        )

        st.markdown(f"##### {t('top_sales_heatmap_title')}")
        st.info(t('top_sales_heatmap_info'))

        if len(top10_pivot.columns) > 31:
            top10_pivot = top10_pivot.iloc[:, -31:]
            st.caption(t('top_sales_heatmap_caption'))

        styled_pivot = top10_pivot.style.background_gradient(cmap='viridis', axis=None).format("{:.0f}", na_rep="-")
        st.dataframe(styled_pivot, use_container_width=True)
    else:
        st.warning(t("no_data_to_display"))

with tab2:
    st.subheader(t("store_comparison_title"))
    sales_by_store = df_period.groupby(['store_id', 'city'])['sales'].sum().reset_index()

    if selected_store != all_stores_option:
        sales_by_store['Статус'] = np.where(sales_by_store['store_id'] == selected_store, t("store_filter_selected"),
                                            t("store_filter_others"))
        color_discrete_map = {t("store_filter_selected"): '#FF4B4B', t("store_filter_others"): '#1E90FF'}
        color_map_key = 'Статус'
    else:
        color_map_key = 'city'
        color_discrete_map = None

    fig_stores = px.bar(
        sales_by_store.sort_values('sales', ascending=False), x='store_id', y='sales', color=color_map_key,
        title=t("store_comparison_title"), labels={'sales': t("chart_sales_axis"), 'store_id': t("store_id_axis")},
        text='sales', color_discrete_map=color_discrete_map
    )
    fig_stores.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_stores, use_container_width=True)

    st.markdown(f"##### {t('store_comparison_data_title')}")
    display_df_stores = sales_by_store.sort_values('sales', ascending=False)
    if 'Статус' in display_df_stores.columns:
        display_df_stores = display_df_stores.drop(columns=['Статус'])

    column_translation = {'store_id': t('store_id_axis'), 'city': t('city_col'), 'sales': t('sales_col')}
    st.dataframe(display_df_stores.rename(columns=column_translation))

    csv = display_df_stores.to_csv(index=False).encode('utf-8')
    st.download_button(t("download_csv_button"), csv, f"store_comparison_{period_days}days.csv", "text/csv")