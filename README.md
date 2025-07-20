<h1 align="center">Aurorix.Retail - Interactive Dashboard</h1>

<p align="center">
  <a href="https://aurorixai-mvp.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
  <img src="https://img.shields.io/badge/status-live-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python">
</p>
### 🏛️ Project Architecture

This project consists of two main components that work together:

1.  **Frontend (This Repository):** An interactive Streamlit dashboard that serves as the user interface for analytics and forecasting.
2.  **Backend (Separate Repository):** A Flask-based REST API that handles all machine learning computations, feature engineering, and model predictions.

➡️ **You can find the backend API repository here: [github.com/arslan15114/aurorix-api](https://github.com/arslan15114/aurorix-api)**

Welcome to the **Aurorix.Retail Dashboard**—an intuitive, multi-page web application designed to transform retail management from guesswork into a data-driven science. Built with Streamlit, this platform provides managers and analysts with a powerful toolkit for making smarter, faster, and more profitable decisions.

The application communicates with our dedicated `aurorix-api` to fetch real-time forecasts and visualizes both predictive and historical data in a user-friendly and actionable way, turning complex data streams into clear business insights.

**➡️ Live Demo:** **[aurorixai-mvp.streamlit.app](https://aurorixai-mvp.streamlit.app/)**

---

## ✨ Key Features

- **🔮 Main Dashboard:** A central command center displaying key performance indicators (KPIs) like model accuracy and analysis speed. It features a detailed, interactive chart for sales dynamics for any selected product/store combination.
- **📈 Historical Analysis Module:** A strategic tool for a 30,000-foot view of the business, featuring:
    - **Top Sales Analysis:** An intuitive heatmap-style table that visualizes not only *what* the top-10 products are, but also *when* they sell best.
    - **Store Comparison:** A benchmarking tool to compare the performance of all retail locations and identify leaders and laggards.
- **🤖 AI Assistant:** An interactive simulation of an AI analyst that acts as a "virtual consultant." It automatically flags key events in the sales history and provides clear, human-readable explanations for anomalies with a single click.
- **🌍 Multi-Language Support:** A fully localized interface with seamless, on-the-fly switching between Russian and Uzbek, demonstrating our readiness for regional scaling.

---

### 🧠 Machine Learning Methodology

The core of our product is a predictive model built using the **LightGBM** gradient boosting algorithm, chosen for its speed and accuracy. To achieve a high level of forecast precision, we developed a comprehensive feature engineering pipeline that turns raw data into intelligent signals for the model.

#### **Key Feature Groups:**
- **Lag Features:** Sales from the previous 1, 7, 14, and 28 days to capture short-term momentum and weekly seasonality.
- **Rolling Statistics:** Moving averages, standard deviations, and other metrics for sales over various time windows to understand recent trends.
- **Calendar Features:** Day of the week, week of the year, month, and flags for national holidays and important pre-holiday periods.
- **Price-Related Features:** Current price, discount percentage, and promotion flags to model the impact of pricing strategies on demand.

The model was trained on a rich historical dataset and underwent a rigorous hyperparameter tuning phase to ensure optimal performance on a validation set.

---

## 🚀 Getting Started (Local Setup)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/arslan15114/aurorix-dashboard.git](https://github.com/arslan15114/aurorix-dashboard.git)
    cd aurorix-dashboard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run streamlit_app.py
    ```
