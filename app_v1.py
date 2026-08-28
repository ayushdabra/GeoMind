import json
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Edinburgh Infection Rate Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------
# 1. LOAD & CACHE DATASETS
# -------------------------------------------------------------
@st.cache_data
def load_all_data():
    base_path = "data/"

    gdf_zones = gpd.read_file(base_path + "edinburgh_iz_boundaries.geojson")
    df_retro = pd.read_csv(base_path + "retrospective_predictions.csv")
    df_future = pd.read_csv(base_path + "future_forecast_20230304.csv")
    df_shapley = pd.read_csv(base_path + "geoshapley.csv")
    df_alpha = pd.read_csv(base_path + "rolling_alpha.csv")
    df_dates = pd.read_csv(base_path + "date_selector.csv")

    with open(base_path + "model_metadata.json", "r") as f:
        metadata = json.load(f)

    return (
        gdf_zones,
        df_retro,
        df_future,
        df_shapley,
        df_alpha,
        df_dates,
        metadata,
    )


(
    gdf_zones,
    df_retro,
    df_future,
    df_shapley,
    df_alpha,
    df_dates,
    metadata,
) = load_all_data()

# -------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("Forecast Controls")

# Mode Switch: Retrospective vs. 4 March Unverified Extrapolation
app_mode = st.sidebar.radio(
    "Select Mode:",
    [
        "Retrospective Evaluation (264 Days)",
        "Unverified Extrapolation (4 March 2023)",
    ],
)

if app_mode == "Retrospective Evaluation (264 Days)":
    date_list = df_dates["target_report_date"].tolist()
    selected_date = st.sidebar.select_slider(
        "Select Target Report Date:",
        options=date_list,
        value=date_list[-1],
    )
    # Filter dataset for selected date
    df_current = df_retro[df_retro["target_report_date"] == selected_date]
    is_extrapolation = False

    # Metric to map
    map_metric = st.sidebar.selectbox(
        "Map Display Metric:",
        options=[
            "predicted_rate",
            "observed_rate",
            "model_absolute_error",
            "predicted_sigma",
        ],
        format_func=lambda x: {
            "predicted_rate": "Predicted Rate (per 100k)",
            "observed_rate": "Observed Rate (Ground Truth)",
            "model_absolute_error": "Absolute Error (|Pred - Obs|)",
            "predicted_sigma": "Uncertainty (Sigma σ)",
        }[x],
    )
else:
    selected_date = "2023-03-04"
    df_current = df_future.copy()
    is_extrapolation = True
    st.sidebar.warning(
        "⚠️ Unverified Extrapolation: No ground truth exists for this date."
    )

    map_metric = st.sidebar.selectbox(
        "Map Display Metric:",
        options=["predicted_rate", "predicted_sigma"],
        format_func=lambda x: {
            "predicted_rate": "Predicted Rate (per 100k)",
            "predicted_sigma": "Uncertainty (Sigma σ)",
        }[x],
    )

# -------------------------------------------------------------
# 3. MAIN DASHBOARD LAYOUT
# -------------------------------------------------------------
st.title("Edinburgh 7-Day Ahead Infection Rate Forecast")
st.caption(
    f"Current View: **{selected_date}** | Update: **{df_current['update_id'].iloc[0]}**"
)

col_map, col_details = st.columns([3, 2])

# Join geospatial features with current data slice
gdf_map = gdf_zones.merge(df_current, on="iz_code")

with col_map:
    st.subheader(f"Spatial Distribution: {map_metric}")

    # Plotly interactive Mapbox Choropleth
    fig_map = px.choropleth_map(
    gdf_map,
    geojson=gdf_map.geometry,
    locations=gdf_map.index,
    color=map_metric,
    color_continuous_scale="Viridis",
    map_style="carto-positron",  # Note: map_style instead of mapbox_style
    zoom=10.5,
    center={"lat": 55.9533, "lon": -3.1883},
    opacity=0.7,
    hover_name="iz_code",
    hover_data={
        "predicted_rate": ":.1f",
        "predicted_sigma": ":.2f",
        "update_id": True,
    },
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=550)
st.plotly_chart(fig_map, use_container_width=True)

with col_details:
    st.subheader("Neighbourhood Inspection")

    # Zone selection dropdown
    iz_list = sorted(gdf_map["iz_code"].unique())
    selected_iz = st.selectbox("Select Intermediate Zone (IZ):", iz_list)

    zone_row = df_current[df_current["iz_code"] == selected_iz].iloc[0]

    # Metrics Summary Cards
    m1, m2 = st.columns(2)
    m1.metric("Predicted Rate", f"{zone_row['predicted_rate']:.1f} / 100k")
    m2.metric(
        "Uncertainty (σ)",
        f"{zone_row['predicted_sigma']:.2f}",
        delta="HIGH" if zone_row.get("uncertainty_flag") == "high" else "NORMAL",
        delta_color="inverse"
        if zone_row.get("uncertainty_flag") == "high"
        else "off",
    )

    if not is_extrapolation:
        m3, m4 = st.columns(2)
        m3.metric("Observed Truth", f"{zone_row['observed_rate']:.1f} / 100k")
        m4.metric("Model Error", f"{zone_row['model_error']:+.1f}")

    # Interval Plot
    st.markdown("##### 80% & 95% Calibrated Confidence Bands")
    fig_ci = go.Figure()
    fig_ci.add_trace(
        go.Bar(
            y=["Forecast Interval"],
            x=[zone_row["calibrated95_upper"] - zone_row["calibrated95_lower"]],
            base=[zone_row["calibrated95_lower"]],
            orientation="h",
            name="95% Interval",
            marker=dict(color="rgba(100, 150, 240, 0.3)"),
        )
    )
    fig_ci.add_trace(
        go.Bar(
            y=["Forecast Interval"],
            x=[zone_row["calibrated80_upper"] - zone_row["calibrated80_lower"]],
            base=[zone_row["calibrated80_lower"]],
            orientation="h",
            name="80% Interval",
            marker=dict(color="rgba(100, 150, 240, 0.6)"),
        )
    )
    fig_ci.add_trace(
        go.Scatter(
            y=["Forecast Interval"],
            x=[zone_row["predicted_rate"]],
            mode="markers",
            name="Predicted",
            marker=dict(color="black", size=12, symbol="diamond"),
        )
    )
    if not is_extrapolation:
        fig_ci.add_trace(
            go.Scatter(
                y=["Forecast Interval"],
                x=[zone_row["observed_rate"]],
                mode="markers",
                name="Observed Truth",
                marker=dict(color="red", size=12, symbol="circle"),
            )
        )
    fig_ci.update_layout(
        barmode="overlay", height=180, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_ci, use_container_width=True)

# -------------------------------------------------------------
# 4. EXPLANATION & RESEARCH TABS (BOTTOM)
# -------------------------------------------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(
    ["GeoShapley Explanation", "Graph Weights (α)", "About & Methodology"]
)

with tab1:
    if is_extrapolation:
        st.subheader(f"GeoShapley Explanation for {selected_iz} (4 March 2023)")
        df_zone_shap = df_shapley[df_shapley["iz_code"] == selected_iz]

        fig_waterfall = go.Figure(
            go.Waterfall(
                name="Shapley",
                orientation="v",
                measure=["relative"] * (len(df_zone_shap) - 1) + ["total"],
                x=df_zone_shap["feature_name"],
                y=df_zone_shap["shapley_value"],
                textposition="outside",
            )
        )
        fig_waterfall.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_waterfall, use_container_width=True)
    else:
        st.info(
            "GeoShapley local explanations are only available for the 4 March 2023 update (U10)."
        )

with tab2:
    st.subheader("Dynamic Graph Weights (α) Across Rolling Updates (U01–U10)")
    fig_alpha = px.line(
        df_alpha,
        x="update_id",
        y=["alpha_geo", "alpha_transport", "alpha_mobility"],
        labels={"value": "Weight (Sums to 1.0)", "update_id": "Model Update"},
        markers=True,
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

with tab3:
    st.markdown("### Model Metadata & Disclaimers")
    st.json(metadata)