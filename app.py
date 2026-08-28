"""
app.py: Real Data Interactive Streamlit Dashboard with ./data configuration.
"""
import os
import json
import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agent import PlanningAgent

# -------------------------------------------------------------
# 1. PAGE SETUP & DATA LOADING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Edinburgh Health Planning Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "./data"

@st.cache_data
def load_base_data():
    gdf_zones = gpd.read_file(os.path.join(DATA_DIR, "edinburgh_iz_boundaries.geojson"))
    if gdf_zones.crs is not None and gdf_zones.crs.to_epsg() != 4326:
        gdf_zones = gdf_zones.to_crs(epsg=4326)
    
    df_dates = pd.read_csv(os.path.join(DATA_DIR, "date_selector.csv"))
    
    shapley_path = os.path.join(DATA_DIR, "geoshapley.csv")
    df_shapley = pd.read_csv(shapley_path) if os.path.exists(shapley_path) else pd.DataFrame()
    
    alpha_path = os.path.join(DATA_DIR, "rolling_alpha.csv")
    df_alpha = pd.read_csv(alpha_path) if os.path.exists(alpha_path) else pd.DataFrame()
    
    meta_path = os.path.join(DATA_DIR, "model_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return gdf_zones, df_dates, df_shapley, df_alpha, metadata

gdf_zones, df_dates, df_shapley, df_alpha, metadata = load_base_data()

@st.cache_resource
def get_agent():
    return PlanningAgent()

agent = get_agent()

# -------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🎯 Planning Controls")

study_area = st.sidebar.text_input("Study Area", value="City of Edinburgh")

# Date selector: 4 March at top, then retrospective dates descending
date_options = ["2023-03-04 (Extrapolation)"] + df_dates["target_report_date"].tolist()[::-1]
chosen_date_label = st.sidebar.selectbox("Forecast Target Date", date_options)
selected_date = chosen_date_label.split(" ")[0]

scenario = st.sidebar.selectbox(
    "Planning Scenario",
    options=["balanced", "coverage priority", "equity priority", "preventive priority"]
)

travel_mode = st.sidebar.radio("Travel Mode", options=["walk", "drive"], index=0)

travel_threshold = st.sidebar.slider(
    "Travel-Time Threshold (min)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=1.0
)

eligible_types = st.sidebar.multiselect(
    "Eligible Facility Types",
    options=["gp", "pharmacy", "mobile_stop"],
    default=["gp", "pharmacy", "mobile_stop"]
)

priority_pop = st.sidebar.selectbox(
    "Priority Population",
    options=["Total Population", "Elderly (65+)", "Deprived Quintiles (SIMD1-2)"]
)

# priority_pop = st.sidebar.selectbox(
#     "Priority Population",
#     options=["Total Population", "Elderly (65+)", "Deprived Quintiles (SIMD1-2)"],
#     index=0,
#     disabled=True,
#     help="Disabled until census demographic datasets (SIMD / age breakdowns) are loaded.",
# )

st.sidebar.info("Constraint: Exactly **6 sites** allocated deterministically.")
run_btn = st.sidebar.button("🚀 Run Agent Pipeline", type="primary")

# -------------------------------------------------------------
# 3. PIPELINE EXECUTION
# -------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if run_btn:
    with st.spinner("Agent evaluating parameters, forecast retrieval, and spatial optimization..."):
        res = agent.run_planning_pipeline(
            study_area=study_area,
            forecast_date=selected_date,
            scenario=scenario,
            travel_mode=travel_mode,
            travel_time_threshold=travel_threshold,
            eligible_site_types=eligible_types,
            priority_population=priority_pop
        )
        st.session_state.pipeline_result = res

res = st.session_state.pipeline_result

# -------------------------------------------------------------
# 4. DASHBOARD PRESENTATION LAYER
# -------------------------------------------------------------
st.title("📍 Edinburgh Infection Forecasting & Facility Allocation")

if res is None:
    st.info("Configure planning parameters in the sidebar and click **'Run Agent Pipeline'**.")
elif not res["success"]:
    st.error(f"🛑 **Pipeline Paused:** {res['message']}")
else:
    diag = res["allocation"]["diagnostics"]
    sites = res["allocation"]["selected_sites"]
    iz_map = res["allocation"]["iz_assignments"]
    forecast = res["forecast"]

    # Status Notification Banner
    if forecast["is_unverified"]:
        st.warning("⚠️ **4 March 2023:** Unverified 7-day extrapolation using checkpoint U10. No ground truth available.")
    else:
        st.success(
            f"✅ Checkpoint `{forecast['checkpoint_used']}` verified. "
            f"Graph Weights: α_geo={forecast['alpha_weights']['alpha_geo']:.2f}, "
            f"α_transport={forecast['alpha_weights']['alpha_transport']:.2f}, "
            f"α_mobility={forecast['alpha_weights']['alpha_mobility']:.2f}"
        )

    # Core Indicator KPI Cards
    st.markdown("### 📊 Planning Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Covered Population", f"{diag['covered_population']:,}")
    k2.metric("IZs Covered", f"{diag['covered_zones']} / {diag['total_zones']}", f"{diag['coverage_percentage']}%")
    k3.metric("Mean Travel Time", f"{diag['mean_travel_time_min']} min")
    k4.metric("Max Travel Time", f"{diag['max_travel_time_min']} min")
    k5.metric("Unserved Zones", f"{diag['unserved_zones']} IZs", delta_color="inverse")

    st.divider()

    # Main Spatial Map and Site Inspector View
    col_map, col_details = st.columns([3, 2])

    with col_map:
        st.subheader("Spatial Risk Distribution & 6 Allocated Sites")
        
        # Prepare GeoDataFrame with dynamic assignment properties
        gdf_plot = gdf_zones.copy()
        gdf_plot["predicted_risk"] = gdf_plot["iz_code"].map(
            lambda z: forecast["zone_forecasts"].get(z, {}).get("predicted_rate", 0.0)
        )
        gdf_plot["status"] = gdf_plot["iz_code"].map(
            lambda z: "Covered" if iz_map.get(z, {}).get("is_covered") else "Unserved"
        )
        gdf_plot["assigned_site"] = gdf_plot["iz_code"].map(
            lambda z: str(iz_map.get(z, {}).get("assigned_site_id", "Unserved"))
        )
        gdf_plot["travel_time"] = gdf_plot["iz_code"].map(
            lambda z: iz_map.get(z, {}).get("travel_time_min", 999.0)
        )

        # Plotly Choropleth map
        if hasattr(px, "choropleth_map"):
            fig = px.choropleth_map(
                gdf_plot,
                geojson=gdf_plot.geometry,
                locations=gdf_plot.index,
                color="predicted_risk",
                color_continuous_scale="Viridis",
                map_style="carto-positron",
                zoom=10.2,
                center={"lat": 55.935, "lon": -3.220},
                opacity=0.65,
                hover_name="iz_code",
                hover_data={"predicted_risk": ":.1f", "status": True, "travel_time": True, "assigned_site": True}
            )
        else:
            fig = px.choropleth_mapbox(
                gdf_plot,
                geojson=gdf_plot.geometry,
                locations=gdf_plot.index,
                color="predicted_risk",
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                zoom=10.2,
                center={"lat": 55.935, "lon": -3.220},
                opacity=0.65,
                hover_name="iz_code",
                hover_data={"predicted_risk": ":.1f", "status": True, "travel_time": True, "assigned_site": True}
            )

        # Convert candidate site locations to WGS84 for pin overlay
        gdf_selected_pts = gpd.GeoDataFrame(
            sites,
            geometry=gpd.points_from_xy([s["easting"] for s in sites], [s["northing"] for s in sites]),
            crs="EPSG:27700"
        ).to_crs(epsg=4326)

        site_layer = go.Scattermap(
            lat=gdf_selected_pts.geometry.y,
            lon=gdf_selected_pts.geometry.x,
            mode="markers+text",
            marker=dict(size=14, color="#D90429"),
            text=[s["site_id"] for s in sites],
            textposition="top right",
            name="6 Selected Sites",
            hovertext=[f"{s['site_name']} ({s['site_type']})" for s in sites]
        ) if hasattr(go, "Scattermap") else go.Scattermapbox(
            lat=gdf_selected_pts.geometry.y,
            lon=gdf_selected_pts.geometry.x,
            mode="markers+text",
            marker=dict(size=14, color="#D90429"),
            text=[s["site_id"] for s in sites],
            textposition="top right",
            name="6 Selected Sites",
            hovertext=[f"{s['site_name']} ({s['site_type']})" for s in sites]
        )

        fig.add_trace(site_layer)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=530)
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.subheader("Selected Site Inspection")
        site_choice = st.selectbox(
            "Inspect Allocated Site:",
            options=[s["site_id"] for s in sites],
            format_func=lambda sid: f"{sid} - {next(s['site_name'] for s in sites if s['site_id'] == sid)}"
        )
        
        st.markdown(agent.explain_site_selection(site_choice, res))

    # Bottom Auxiliary Tabs
    st.divider()
    tab_comp, tab_shap, tab_alpha, tab_audit = st.tabs([
        "⚖️ Policy Comparison", "🔍 GeoShapley (4 March)", "📈 Alpha Weights (U01–U10)", "📋 Agent Audit Log"
    ])

    with tab_comp:
        st.markdown("##### 4 Planning Scenarios Evaluated Under Same Constraints")
        st.dataframe(res["comparison_df"], use_container_width=True, hide_index=True)

    with tab_shap:
        if forecast["is_unverified"] and not df_shapley.empty:
            st.markdown("##### Local Explanations for 4 March 2023 (Checkpoint U10)")
            iz_choice = st.selectbox("Select Zone for GeoShapley Breakdown:", sorted(list(forecast["zone_forecasts"].keys())))
            df_iz_shap = df_shapley[df_shapley["iz_code"] == iz_choice]
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Shapley",
                orientation="v",
                measure=["relative"] * (len(df_iz_shap) - 1) + ["total"],
                x=df_iz_shap["feature_name"],
                y=df_iz_shap["shapley_value"],
                textposition="outside"
            ))
            fig_waterfall.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            st.info("GeoShapley decomposition is only available for the 4 March 2023 checkpoint (U10).")

    with tab_alpha:
        if not df_alpha.empty:
            st.markdown("##### Dynamic Graph Fusion Weights Across Test Updates")
            fig_alpha = px.line(
                df_alpha,
                x="update_id",
                y=["alpha_geo", "alpha_transport", "alpha_mobility"],
                markers=True
            )
            st.plotly_chart(fig_alpha, use_container_width=True)

    with tab_audit:
        st.markdown("##### Agent Tool-Calling Trace")
        for log in res["logs"]:
            st.code(f"[{log['step']}]\n{json.dumps(log['details'], indent=2)}")