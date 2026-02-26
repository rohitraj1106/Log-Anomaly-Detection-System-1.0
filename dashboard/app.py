"""
Streamlit Dashboard — Log Anomaly Detection Visualization.

Provides interactive dashboards showing:
- Anomaly spikes over time
- Service-wise anomaly rates
- Log level distributions
- Top anomalous patterns
- Model performance metrics
- Real-time monitoring metrics

Run: streamlit run dashboard/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Log Anomaly Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
    }
    .alert-critical { color: #ff4444; font-weight: bold; }
    .alert-warning { color: #ffbb33; font-weight: bold; }
    .status-healthy { color: #00C851; }
    .stMetric label { font-size: 0.9rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data(ttl=30)
def load_log_data() -> pd.DataFrame:
    """Load processed log data for visualization."""
    csv_path = Path(PROJECT_ROOT) / "data" / "raw" / "system_logs.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    # Generate sample data if no file exists
    return _generate_demo_data()


@st.cache_data(ttl=30)
def load_evaluation_report() -> dict:
    """Load the latest evaluation report."""
    eval_dir = Path(PROJECT_ROOT) / "experiments" / "evaluations"
    if eval_dir.exists():
        reports = list(eval_dir.glob("*.json"))
        if reports:
            latest = sorted(reports)[-1]
            with open(latest) as f:
                return json.load(f)
    return {}


@st.cache_data(ttl=30)
def load_monitoring_metrics() -> dict:
    """Load monitoring metrics."""
    metrics_dir = Path(PROJECT_ROOT) / "monitoring" / "metrics"
    if metrics_dir.exists():
        files = list(metrics_dir.glob("*.json"))
        if files:
            latest = sorted(files)[-1]
            with open(latest) as f:
                return json.load(f)
    return {}


def _generate_demo_data() -> pd.DataFrame:
    """Generate demo data for dashboard preview."""
    np.random.seed(42)
    n = 5000
    services = [
        "auth-service",
        "api-gateway",
        "payment-service",
        "user-service",
        "notification-service",
        "search-service",
    ]
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    timestamps = pd.date_range(end=datetime.now(), periods=n, freq="1min")

    data = {
        "timestamp": timestamps,
        "level": np.random.choice(levels, n, p=[0.1, 0.55, 0.15, 0.15, 0.05]),
        "service": np.random.choice(services, n),
        "message": [f"Sample log message {i}" for i in range(n)],
        "is_anomaly": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    }
    return pd.DataFrame(data)


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.markdown("## 🔍 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    [
        "📊 Overview",
        "🔴 Anomaly Analysis",
        "🏗️ Service Health",
        "📈 Model Performance",
        "⚡ Real-Time Monitor",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Filters")

# Load data
df = load_log_data()

# Date range filter
if "timestamp" in df.columns and not df["timestamp"].isna().all():
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            mask = (df["timestamp"].dt.date >= date_range[0]) & (
                df["timestamp"].dt.date <= date_range[1]
            )
            df = df[mask]

# Service filter
if "service" in df.columns:
    services = ["All", *sorted(df["service"].unique().tolist())]
    selected_service = st.sidebar.selectbox("Service", services)
    if selected_service != "All":
        df = df[df["service"] == selected_service]


# =============================================================================
# Page: Overview
# =============================================================================
if page == "📊 Overview":
    st.markdown(
        '<div class="main-header">Log Anomaly Detection Platform</div>', unsafe_allow_html=True
    )
    st.markdown("Real-time monitoring and anomaly detection for distributed system logs.")

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_logs = len(df)
    anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    anomaly_rate = anomaly_count / max(total_logs, 1)
    unique_services = df["service"].nunique() if "service" in df.columns else 0
    error_count = len(df[df["level"].isin(["ERROR", "CRITICAL"])]) if "level" in df.columns else 0

    col1.metric("Total Logs", f"{total_logs:,}")
    col2.metric("Anomalies Detected", f"{anomaly_count:,}", delta=f"{anomaly_rate:.1%}")
    col3.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
    col4.metric("Services Monitored", unique_services)
    col5.metric("Errors/Critical", f"{error_count:,}")

    st.markdown("---")

    # Timeline chart
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Log Volume Over Time")
        if "timestamp" in df.columns:
            hourly = df.set_index("timestamp").resample("1h").size().reset_index()
            hourly.columns = ["timestamp", "count"]

            fig = px.area(
                hourly,
                x="timestamp",
                y="count",
                color_discrete_sequence=["#667eea"],
                labels={"count": "Log Count", "timestamp": "Time"},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Log Level Distribution")
        if "level" in df.columns:
            level_counts = df["level"].value_counts()
            colors = {
                "DEBUG": "#36a2eb",
                "INFO": "#4bc0c0",
                "WARNING": "#ffce56",
                "ERROR": "#ff6384",
                "CRITICAL": "#ff0000",
            }
            fig = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                color=level_counts.index,
                color_discrete_map=colors,
                hole=0.4,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Recent anomalies table
    st.subheader("🚨 Recent Anomalies")
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1].tail(10).sort_values("timestamp", ascending=False)
        if not anomalies.empty:
            display_cols = [
                c for c in ["timestamp", "level", "service", "message"] if c in anomalies.columns
            ]
            st.dataframe(anomalies[display_cols], use_container_width=True, height=300)
        else:
            st.info("No anomalies detected in the selected range.")


# =============================================================================
# Page: Anomaly Analysis
# =============================================================================
elif page == "🔴 Anomaly Analysis":
    st.markdown("## 🔴 Anomaly Analysis")

    if "is_anomaly" in df.columns and "timestamp" in df.columns:
        # Anomaly timeline
        st.subheader("Anomaly Spikes Over Time")
        anomaly_ts = (
            df[df["is_anomaly"] == 1].set_index("timestamp").resample("1h").size().reset_index()
        )
        anomaly_ts.columns = ["timestamp", "anomaly_count"]
        normal_ts = (
            df[df["is_anomaly"] == 0].set_index("timestamp").resample("1h").size().reset_index()
        )
        normal_ts.columns = ["timestamp", "normal_count"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=normal_ts["timestamp"],
                y=normal_ts["normal_count"],
                name="Normal",
                fill="tozeroy",
                line={"color": "#4bc0c0"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomaly_ts["timestamp"],
                y=anomaly_ts["anomaly_count"],
                name="Anomaly",
                fill="tozeroy",
                line={"color": "#ff6384"},
            )
        )
        fig.update_layout(
            title="Normal vs Anomalous Log Volume",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pattern analysis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Anomalous Services")
            if "service" in df.columns:
                service_anomalies = df[df["is_anomaly"] == 1]["service"].value_counts().head(10)
                fig = px.bar(
                    x=service_anomalies.index,
                    y=service_anomalies.values,
                    color=service_anomalies.values,
                    color_continuous_scale="Reds",
                    labels={"x": "Service", "y": "Anomaly Count"},
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Anomaly by Log Level")
            if "level" in df.columns:
                level_anomalies = df[df["is_anomaly"] == 1]["level"].value_counts()
                fig = px.bar(
                    x=level_anomalies.index,
                    y=level_anomalies.values,
                    color=level_anomalies.index,
                    color_discrete_map={
                        "DEBUG": "#36a2eb",
                        "INFO": "#4bc0c0",
                        "WARNING": "#ffce56",
                        "ERROR": "#ff6384",
                        "CRITICAL": "#ff0000",
                    },
                    labels={"x": "Level", "y": "Count"},
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Top anomalous patterns
        st.subheader("Top Anomalous Log Patterns")
        if "message" in df.columns:
            anomalous_msgs = df[df["is_anomaly"] == 1]["message"].value_counts().head(15)
            for i, (msg, count) in enumerate(anomalous_msgs.items()):
                st.markdown(f"**{i + 1}.** `{str(msg)[:120]}` — **{count}** occurrences")
    else:
        st.info("No anomaly labels available. Run the training pipeline first.")


# =============================================================================
# Page: Service Health
# =============================================================================
elif page == "🏗️ Service Health":
    st.markdown("## 🏗️ Service Health Dashboard")

    if "service" in df.columns:
        services = df["service"].unique()

        # Service health grid
        cols = st.columns(min(3, len(services)))
        for i, service in enumerate(sorted(services)):
            service_df = df[df["service"] == service]
            total = len(service_df)
            anomalies = (
                int(service_df["is_anomaly"].sum()) if "is_anomaly" in service_df.columns else 0
            )
            rate = anomalies / max(total, 1)

            health = "🟢" if rate < 0.05 else "🟡" if rate < 0.15 else "🔴"

            with cols[i % len(cols)]:
                st.markdown(f"### {health} {service}")
                st.metric("Total Logs", f"{total:,}")
                st.metric("Anomalies", f"{anomalies:,}", delta=f"{rate:.1%}")

        # Service comparison
        st.markdown("---")
        st.subheader("Service-wise Anomaly Rate Comparison")
        if "is_anomaly" in df.columns:
            service_stats = (
                df.groupby("service")
                .agg(
                    total=("is_anomaly", "count"),
                    anomalies=("is_anomaly", "sum"),
                )
                .reset_index()
            )
            service_stats["anomaly_rate"] = service_stats["anomalies"] / service_stats["total"]

            fig = px.bar(
                service_stats.sort_values("anomaly_rate", ascending=False),
                x="service",
                y="anomaly_rate",
                color="anomaly_rate",
                color_continuous_scale="RdYlGn_r",
                labels={"anomaly_rate": "Anomaly Rate", "service": "Service"},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Model Performance
# =============================================================================
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")

    report = load_evaluation_report()

    if report:
        # Classification metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", f"{report.get('precision', 0):.4f}")
        col2.metric("Recall", f"{report.get('recall', 0):.4f}")
        col3.metric("F1 Score", f"{report.get('f1_score', 0):.4f}")
        col4.metric("ROC-AUC", f"{report.get('roc_auc', 'N/A')}")

        # Confusion Matrix
        cm = report.get("confusion_matrix", {})
        if cm:
            st.subheader("Confusion Matrix")
            cm_array = np.array(
                [
                    [cm.get("true_negatives", 0), cm.get("false_positives", 0)],
                    [cm.get("false_negatives", 0), cm.get("true_positives", 0)],
                ]
            )
            fig = px.imshow(
                cm_array,
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                x=["Normal", "Anomaly"],
                y=["Normal", "Anomaly"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400, font_color="#ccc")
            st.plotly_chart(fig, use_container_width=True)

        # Score distribution
        if "score_distribution" in report:
            st.subheader("Anomaly Score Distribution")
            dist = report["score_distribution"]
            fig = go.Figure(
                go.Bar(
                    x=dist["bin_edges"][:-1],
                    y=dist["counts"],
                    marker_color="#667eea",
                )
            )
            fig.update_layout(
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation reports found. Run the training pipeline to generate reports.")

        # Show demo metrics
        st.markdown("### Demo Metrics (placeholder)")
        demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
        demo_col1.metric("Precision", "0.8723")
        demo_col2.metric("Recall", "0.7891")
        demo_col3.metric("F1 Score", "0.8286")
        demo_col4.metric("ROC-AUC", "0.9142")


# =============================================================================
# Page: Real-Time Monitor
# =============================================================================
elif page == "⚡ Real-Time Monitor":
    st.markdown("## ⚡ Real-Time Monitoring")

    monitoring_data = load_monitoring_metrics()

    if monitoring_data:
        preds = monitoring_data.get("predictions", {})
        anomalies = monitoring_data.get("anomalies", {})
        latency = monitoring_data.get("latency", {})
        errors = monitoring_data.get("errors", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", f"{preds.get('total', 0):,}")
        col2.metric("Anomaly Rate", f"{anomalies.get('window_rate', 0):.2%}")
        col3.metric("Avg Latency", f"{latency.get('mean', 0):.1f}ms")
        col4.metric("Error Rate", f"{errors.get('error_rate', 0):.2%}")

        # Latency breakdown
        st.subheader("Latency Percentiles")
        latency_data = {
            "Percentile": ["P50", "P90", "P95", "P99"],
            "Latency (ms)": [
                latency.get("p50", 0),
                latency.get("p90", 0),
                latency.get("p95", 0),
                latency.get("p99", 0),
            ],
        }
        fig = px.bar(
            latency_data,
            x="Percentile",
            y="Latency (ms)",
            color="Latency (ms)",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monitoring data available. Run the API server to collect metrics.")

        # Simulated real-time metrics
        st.markdown("### Simulated Metrics")
        chart_data = pd.DataFrame(
            np.random.randn(50, 3) * [10, 5, 2] + [100, 50, 5],
            columns=["Predictions/min", "Avg Latency (ms)", "Anomalies/min"],
        )
        st.line_chart(chart_data, height=300)


# =============================================================================
# Footer
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Log Anomaly Detection Platform** v1.0  \nBuilt with ❤️ for FAANG-level ML Engineering"
)
