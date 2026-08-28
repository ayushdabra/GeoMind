"""
tools.py: Deterministic Tools configured for flat ./data directory.
"""
import os
import json
from typing import Dict, Any, List
import pandas as pd
import geopandas as gpd
import numpy as np

DATA_DIR = "./data"

# ---------------------------------------------------------------------------
# 1. TOOL: Model Compatibility Check
# ---------------------------------------------------------------------------
def check_model_compatibility(study_area: str, forecast_date: str) -> Dict[str, Any]:
    area_norm = study_area.strip().lower()
    if "edinburgh" not in area_norm:
        return {
            "status": "incompatible_region",
            "decision": "require_new_region_training",
            "message": f"No pre-trained model exists for '{study_area}'. Full region training pipeline required.",
            "requires_user_confirmation": True,
            "checkpoint_id": None,
            "is_extrapolation": False
        }
    
    if forecast_date == "2023-03-04":
        return {
            "status": "compatible_unverified",
            "decision": "run_inference",
            "checkpoint_id": "U10",
            "message": "Loaded validated checkpoint U10 (Unverified Extrapolation).",
            "requires_user_confirmation": False,
            "is_extrapolation": True
        }
    elif "2022-06-07" <= forecast_date <= "2023-02-25":
        return {
            "status": "compatible_retrospective",
            "decision": "run_inference",
            "checkpoint_id": "U10",
            "message": "Loaded validated retrospective checkpoint.",
            "requires_user_confirmation": False,
            "is_extrapolation": False
        }
    else:
        return {
            "status": "outdated_data_window",
            "decision": "require_rolling_update",
            "message": f"Date '{forecast_date}' is outside the calibrated window. Requires rolling update.",
            "requires_user_confirmation": True,
            "checkpoint_id": None,
            "is_extrapolation": False
        }

# ---------------------------------------------------------------------------
# 2. TOOL: Forecast Inference / Real Data Fetch
# ---------------------------------------------------------------------------
def run_forecast_inference(checkpoint_id: str, forecast_date: str, study_area: str = "City of Edinburgh") -> Dict[str, Any]:
    path_future = os.path.join(DATA_DIR, "future_forecast_20230304.csv")
    path_retro = os.path.join(DATA_DIR, "retrospective_predictions.csv")
    path_alpha = os.path.join(DATA_DIR, "rolling_alpha.csv")

    df_alpha = pd.read_csv(path_alpha)
    
    if forecast_date == "2023-03-04":
        df_forecast = pd.read_csv(path_future)
        alpha_row = df_alpha[df_alpha["update_id"] == "U10"].iloc[0]
        alphas = {
            "alpha_geo": float(alpha_row["alpha_geo"]),
            "alpha_transport": float(alpha_row["alpha_transport"]),
            "alpha_mobility": float(alpha_row["alpha_mobility"]),
        }
        is_unverified = True
    else:
        df_all_retro = pd.read_csv(path_retro)
        df_forecast = df_all_retro[df_all_retro["target_report_date"] == forecast_date].copy()
        if len(df_forecast) == 0:
            raise ValueError(f"Date {forecast_date} not found in retrospective dataset.")
        
        alphas = {
            "alpha_geo": float(df_forecast["alpha_geo"].iloc[0]),
            "alpha_transport": float(df_forecast["alpha_transport"].iloc[0]),
            "alpha_mobility": float(df_forecast["alpha_mobility"].iloc[0]),
        }
        is_unverified = False

    records = df_forecast.to_dict(orient="records")
    return {
        "status": "success",
        "forecast_date": forecast_date,
        "checkpoint_used": checkpoint_id,
        "total_zones": len(records),
        "alpha_weights": alphas,
        "is_unverified": is_unverified,
        "zone_forecasts": {r["iz_code"]: r for r in records}
    }

# ---------------------------------------------------------------------------
# 3. TOOL: Deterministic Location-Allocation Engine
# ---------------------------------------------------------------------------
def run_location_allocation(
    forecast_data: Dict[str, Any],
    scenario: str,
    travel_mode: str,
    travel_time_threshold: float,
    eligible_site_types: List[str],
    priority_population: str,
    fixed_k: int = 6
) -> Dict[str, Any]:
    sites_path = os.path.join(DATA_DIR, "edinburgh_merged_candidate_sites.shp")
    matrix_path = os.path.join(DATA_DIR, "travel_time_matrix.csv")

    gdf_sites = gpd.read_file(sites_path)
    df_matrix = pd.read_csv(matrix_path)

    # Filter candidate sites by user-chosen types
    if eligible_site_types:
        gdf_sites = gdf_sites[gdf_sites["site_type"].isin(eligible_site_types)].copy()

    # Filter matrix by travel mode and eligible candidate sites
    df_mode_matrix = df_matrix[
        (df_matrix["mode"] == travel_mode) & 
        (df_matrix["site_id"].isin(gdf_sites["site_id"]))
    ].copy()

    zone_forecasts = forecast_data["zone_forecasts"]
    iz_codes = list(zone_forecasts.keys())

    # 1. Compute Zone Objective Weights
    iz_weights = {}
    for iz in iz_codes:
        f = zone_forecasts.get(iz, {})
        pred_rate = f.get("predicted_rate", 100.0)
        sigma = f.get("predicted_sigma", 15.0)
        pop = 4000.0
        
        if scenario == "coverage priority":
            w = pop
        elif scenario == "equity priority":
            w = pop * (1.0 + (sigma / 25.0))
        elif scenario == "preventive priority":
            w = pop * (pred_rate / 100.0)
        else:  # balanced
            w = (0.4 * pop) + (0.4 * pop * (pred_rate / 100.0)) + (0.2 * pop * (sigma / 20.0))
        
        iz_weights[iz] = max(1.0, float(w))

    # 2. Greedy Maximum Coverage Solver (MCLP)
    covered_pairs = df_mode_matrix[df_mode_matrix["travel_time_min"] <= travel_time_threshold]
    
    site_coverage_map = {}
    for site_id, group in covered_pairs.groupby("site_id"):
        site_coverage_map[site_id] = set(group["iz_code"])

    selected_site_ids = []
    uncovered_izs = set(iz_codes)
    candidate_id_pool = list(gdf_sites["site_id"].unique())
    
    for _ in range(min(fixed_k, len(candidate_id_pool))):
        best_site = None
        best_score = -1.0
        
        for s_id in candidate_id_pool:
            if s_id in selected_site_ids:
                continue
            covered_zones = site_coverage_map.get(s_id, set()) & uncovered_izs
            score = sum(iz_weights[z] for z in covered_zones)
            if score > best_score:
                best_score = score
                best_site = s_id
        
        if best_site is not None and best_score > 0:
            selected_site_ids.append(best_site)
            uncovered_izs -= site_coverage_map.get(best_site, set())
        else:
            remaining = [s for s in candidate_id_pool if s not in selected_site_ids]
            if remaining:
                selected_site_ids.append(remaining[0])

    # 3. Zone Assignments & Metrics
    gdf_selected = gdf_sites[gdf_sites["site_id"].isin(selected_site_ids)].copy()
    df_selected_matrix = df_mode_matrix[df_mode_matrix["site_id"].isin(selected_site_ids)]
    
    iz_assignment_map = {}
    assigned_counts = {sid: 0 for sid in selected_site_ids}
    travel_times_recorded = []
    covered_iz_count = 0
    total_covered_pop = 0

    for iz in iz_codes:
        iz_times = df_selected_matrix[df_selected_matrix["iz_code"] == iz]
        if len(iz_times) > 0 and not iz_times["travel_time_min"].isna().all():
            min_row = iz_times.sort_values("travel_time_min").iloc[0]
            tt = float(min_row["travel_time_min"])
            assigned_sid = min_row["site_id"]
            
            is_cov = (tt <= travel_time_threshold)
            if is_cov:
                covered_iz_count += 1
                total_covered_pop += int(iz_weights[iz])
                assigned_counts[assigned_sid] += 1
                travel_times_recorded.append(tt)

            iz_assignment_map[iz] = {
                "assigned_site_id": assigned_sid if is_cov else "Unserved",
                "travel_time_min": round(tt, 1),
                "is_covered": is_cov
            }
        else:
            iz_assignment_map[iz] = {
                "assigned_site_id": "Unserved",
                "travel_time_min": 999.0,
                "is_covered": False
            }

    selected_sites_list = []
    for _, row in gdf_selected.iterrows():
        sid = row["site_id"]
        sname = row.get("site_name", f"Facility {sid}")
        stype = row.get("site_type", "facility")
        
        selected_sites_list.append({
            "site_id": sid,
            "site_name": sname,
            "site_type": stype,
            "easting": float(row.geometry.x),
            "northing": float(row.geometry.y),
            "assigned_iz_count": assigned_counts.get(sid, 0),
            "served_population": int(assigned_counts.get(sid, 0) * 4000),
            "selection_reason": f"Optimized for '{scenario}' under {travel_time_threshold} min threshold ({travel_mode})."
        })

    diagnostics = {
        "scenario": scenario,
        "fixed_site_count": len(selected_sites_list),
        "travel_mode": travel_mode,
        "travel_time_threshold_min": travel_time_threshold,
        "total_zones": len(iz_codes),
        "covered_zones": covered_iz_count,
        "unserved_zones": len(iz_codes) - covered_iz_count,
        "covered_population": total_covered_pop,
        "unserved_population": int((len(iz_codes) - covered_iz_count) * 4000),
        "mean_travel_time_min": float(np.mean(travel_times_recorded).round(1)) if travel_times_recorded else 0.0,
        "max_travel_time_min": float(np.max(travel_times_recorded).round(1)) if travel_times_recorded else 0.0,
        "coverage_percentage": round((covered_iz_count / len(iz_codes)) * 100, 1) if iz_codes else 0.0
    }

    return {
        "status": "success",
        "selected_sites": selected_sites_list,
        "iz_assignments": iz_assignment_map,
        "diagnostics": diagnostics
    }

# ---------------------------------------------------------------------------
# 4. TOOL: Scenario Comparator
# ---------------------------------------------------------------------------
def compare_all_scenarios(
    forecast_data: Dict[str, Any],
    travel_mode: str,
    travel_time_threshold: float,
    eligible_site_types: List[str],
    priority_population: str
) -> pd.DataFrame:
    scenarios = ["coverage priority", "equity priority", "preventive priority", "balanced"]
    rows = []
    for sc in scenarios:
        res = run_location_allocation(
            forecast_data=forecast_data,
            scenario=sc,
            travel_mode=travel_mode,
            travel_time_threshold=travel_time_threshold,
            eligible_site_types=eligible_site_types,
            priority_population=priority_population
        )
        d = res["diagnostics"]
        rows.append({
            "Scenario": sc.title(),
            "Covered IZs": f"{d['covered_zones']} / {d['total_zones']}",
            "Coverage %": f"{d['coverage_percentage']}%",
            "Covered Pop": f"{d['covered_population']:,}",
            "Mean Travel Time": f"{d['mean_travel_time_min']} min",
            "Max Travel Time": f"{d['max_travel_time_min']} min",
            "Unserved IZs": d["unserved_zones"]
        })
    return pd.DataFrame(rows)