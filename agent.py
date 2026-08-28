"""
agent.py: Orchestration and Control Layer Agent.
Acts as the intermediary between user inputs, the forecasting pipeline,
the deterministic location-allocation solver, and the UI presentation layer.
"""

from typing import Dict, Any, List, Optional
import json
import tools


class PlanningAgent:
    """
    Control-layer Agent that manages workflow state, compatibility checks,
    tool calls, result validation, and explanation synthesis.
    """

    def __init__(self):
        self.execution_log: List[Dict[str, Any]] = []

    def _log_step(self, step_name: str, payload: Any):
        """Maintains a transparent audit log of deterministic decisions."""
        self.execution_log.append({
            "step": step_name,
            "details": payload
        })

    def validate_inputs(
        self,
        study_area: str,
        forecast_date: str,
        scenario: str,
        travel_mode: str,
        travel_time_threshold: float,
        eligible_site_types: List[str]
    ) -> Optional[str]:
        """Validates incoming planning request parameters against allowed constraints."""
        valid_scenarios = ["coverage priority", "equity priority", "preventive priority", "balanced"]
        if scenario.lower() not in valid_scenarios:
            return f"Invalid scenario '{scenario}'. Must be one of {valid_scenarios}."

        if travel_mode.lower() not in ["walk", "drive"]:
            return f"Invalid travel mode '{travel_mode}'. Supported modes: ['walk', 'drive']."

        if travel_time_threshold <= 0 or travel_time_threshold > 120:
            return f"Travel threshold ({travel_time_threshold} min) must be between 1 and 120 minutes."

        if not eligible_site_types:
            return "At least one candidate site type must be selected."

        return None

    def run_planning_pipeline(
        self,
        study_area: str,
        forecast_date: str,
        scenario: str,
        travel_mode: str,
        travel_time_threshold: float,
        eligible_site_types: List[str],
        priority_population: str,
        user_confirmed_retraining: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the primary end-to-end planning workflow:
        1. Validate Inputs
        2. Compatibility Check (Router)
        3. Forecast Inference / Data Fetch
        4. Location-Allocation Engine
        5. Output Verification
        6. Multi-Scenario Diagnostics
        """
        self.execution_log = []

        # -------------------------------------------------------------
        # 1. INPUT VALIDATION
        # -------------------------------------------------------------
        validation_error = self.validate_inputs(
            study_area=study_area,
            forecast_date=forecast_date,
            scenario=scenario,
            travel_mode=travel_mode,
            travel_time_threshold=travel_time_threshold,
            eligible_site_types=eligible_site_types
        )
        if validation_error:
            self._log_step("0. Input Validation Failed", {"error": validation_error})
            return {
                "success": False,
                "error_type": "INPUT_VALIDATION_ERROR",
                "message": validation_error,
                "logs": self.execution_log
            }

        # -------------------------------------------------------------
        # 2. MODEL COMPATIBILITY & ROUTING
        # -------------------------------------------------------------
        compat_result = tools.check_model_compatibility(study_area, forecast_date)
        self._log_step("1. Compatibility & Routing Check", compat_result)

        if compat_result["requires_user_confirmation"] and not user_confirmed_retraining:
            return {
                "success": False,
                "error_type": "REQUIRES_USER_CONFIRMATION",
                "decision": compat_result["decision"],
                "message": compat_result["message"],
                "logs": self.execution_log
            }

        # -------------------------------------------------------------
        # 3. FORECAST INFERENCE
        # -------------------------------------------------------------
        forecast_result = tools.run_forecast_inference(
            checkpoint_id=compat_result["checkpoint_id"],
            forecast_date=forecast_date,
            study_area=study_area
        )
        self._log_step(
            "2. Forecast Retrieval & Inference",
            {
                "checkpoint": forecast_result["checkpoint_used"],
                "zones_loaded": forecast_result["total_zones"],
                "alphas": forecast_result["alpha_weights"],
                "is_unverified": forecast_result["is_unverified"]
            }
        )

        # -------------------------------------------------------------
        # 4. DETERMINISTIC LOCATION-ALLOCATION (6 SITES)
        # -------------------------------------------------------------
        allocation_result = tools.run_location_allocation(
            forecast_data=forecast_result,
            scenario=scenario,
            travel_mode=travel_mode,
            travel_time_threshold=travel_time_threshold,
            eligible_site_types=eligible_site_types,
            priority_population=priority_population,
            fixed_k=6
        )
        self._log_step(
            "3. Deterministic Location Allocation",
            {
                "sites_allocated": len(allocation_result["selected_sites"]),
                "coverage_pct": allocation_result["diagnostics"]["coverage_percentage"],
                "covered_pop": allocation_result["diagnostics"]["covered_population"]
            }
        )

        # -------------------------------------------------------------
        # 5. POST-EXECUTION VERIFICATION SAFEGUARD
        # -------------------------------------------------------------
        selected_count = len(allocation_result["selected_sites"])
        if selected_count != 6:
            err_msg = f"System Error: Optimization engine returned {selected_count} sites instead of fixed 6."
            self._log_step("4. Verification Failed", {"error": err_msg})
            return {
                "success": False,
                "error_type": "ALLOCATION_SCHEMA_ERROR",
                "message": err_msg,
                "logs": self.execution_log
            }

        # -------------------------------------------------------------
        # 6. MULTI-SCENARIO BENCHMARK COMPARISON
        # -------------------------------------------------------------
        scenario_comparison_df = tools.compare_all_scenarios(
            forecast_data=forecast_result,
            travel_mode=travel_mode,
            travel_time_threshold=travel_time_threshold,
            eligible_site_types=eligible_site_types,
            priority_population=priority_population
        )
        self._log_step("5. Policy Comparison", "Computed side-by-side metrics across all 4 scenarios.")

        return {
            "success": True,
            "compat": compat_result,
            "forecast": forecast_result,
            "allocation": allocation_result,
            "comparison_df": scenario_comparison_df,
            "logs": self.execution_log
        }

    def explain_site_selection(
        self,
        site_id: str,
        pipeline_output: Dict[str, Any]
    ) -> str:
        """
        Generates a strict, non-hallucinated narrative explanation for a selected site
        based entirely on returned solver metrics and diagnostics.
        """
        if not pipeline_output.get("success"):
            return "Cannot generate explanation: The planning pipeline did not complete successfully."

        sites = pipeline_output["allocation"]["selected_sites"]
        diag = pipeline_output["allocation"]["diagnostics"]

        target_site = next((s for s in sites if s["site_id"] == site_id), None)
        if not target_site:
            available_ids = [s["site_id"] for s in sites]
            return f"Diagnostic Error: Site ID '{site_id}' is not in the active allocation set: {available_ids}."

        explanation = (
            f"### 📋 Site Selection Audit: {target_site['site_name']} (`{target_site['site_id']}`)\n\n"
            f"- **Facility Type:** {target_site['site_type'].upper()}\n"
            f"- **Intermediate Zones Assigned:** {target_site['assigned_iz_count']} zones\n"
            f"- **Estimated Population Served:** ~{target_site['served_population']:,} residents\n"
            f"- **Active Policy:** {diag['scenario'].title()} optimization\n"
            f"- **Mobility Constraints:** {diag['travel_time_threshold_min']} min threshold ({diag['travel_mode']})\n"
            f"- **Mathematical Basis:** {target_site['selection_reason']}\n\n"
            f"> **System Safeguard:** All allocations are computed deterministically via network distance matrices and objective function weights. The language model does not choose, alter, or prioritize locations independently."
        )
        return explanation