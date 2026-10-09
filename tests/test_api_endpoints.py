"""
API regression tests for the T1D Digital Twin FastAPI backend.

Uses FastAPI TestClient (no running server needed).
Covers: health, patients, overview, trace, experiments, whatif, privacy.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def first_pid(client):
    resp = client.get("/api/patients")
    assert resp.status_code == 200
    pts = resp.json()["patients"]
    assert len(pts) > 0
    return pts[0]["id"]


class TestHealth:
    def test_200(self, client):
        assert client.get("/api/health").status_code == 200

    def test_schema(self, client):
        d = client.get("/api/health").json()
        for k in ["status", "api", "dataset_ready", "patient_count", "mechanistic_model", "disclaimer"]:
            assert k in d

    def test_dataset_ready(self, client):
        d = client.get("/api/health").json()
        assert d["dataset_ready"] is True
        assert d["patient_count"] >= 1

    def test_model_available(self, client):
        assert client.get("/api/health").json()["mechanistic_model"] == "available"


class TestPatients:
    def test_200(self, client):
        assert client.get("/api/patients").status_code == 200

    def test_nonempty(self, client):
        d = client.get("/api/patients").json()
        assert d["count"] >= 1

    def test_entry_schema(self, client):
        for p in client.get("/api/patients").json()["patients"]:
            for k in ["id", "label", "data_origin", "mean_glucose_mgdL", "tir_pct", "n_readings"]:
                assert k in p
            assert p["data_origin"] == "synthetic"
            assert p["n_readings"] > 0

    def test_files_exist(self, client):
        for p in client.get("/api/patients").json()["patients"]:
            fpath = REPO_ROOT / "data" / "processed" / f"patient_{p['id']}.parquet"
            assert fpath.exists(), f"Missing: {fpath}"


class TestOverview:
    def test_200(self, client, first_pid):
        assert client.get(f"/api/overview/{first_pid}?window_hours=24").status_code == 200

    def test_404(self, client):
        assert client.get("/api/overview/nope_xyz?window_hours=24").status_code == 404

    def test_schema(self, client, first_pid):
        d = client.get(f"/api/overview/{first_pid}?window_hours=24").json()
        for k in ["patient_id", "n_readings", "current_glucose_mgdL", "trend_arrow",
                  "metrics", "cgm_trace", "meal_events", "forecast", "disclaimer"]:
            assert k in d

    def test_metrics_sum(self, client, first_pid):
        m = client.get(f"/api/overview/{first_pid}?window_hours=24").json()["metrics"]
        total = m["tir_pct"] + m["tbr_pct"] + m["tar_pct"]
        assert abs(total - 100.0) < 0.6

    def test_cgm_trace_nonempty(self, client, first_pid):
        d = client.get(f"/api/overview/{first_pid}?window_hours=24").json()
        assert len(d["cgm_trace"]) > 0
        for pt in d["cgm_trace"][:3]:
            assert "t" in pt and "glucose_mgdL" in pt
            assert 20 <= pt["glucose_mgdL"] <= 600

    def test_window_filtering(self, client, first_pid):
        n6 = client.get(f"/api/overview/{first_pid}?window_hours=6").json()["n_readings"]
        n24 = client.get(f"/api/overview/{first_pid}?window_hours=24").json()["n_readings"]
        assert n24 >= n6

    def test_forecast_present(self, client, first_pid):
        d = client.get(f"/api/overview/{first_pid}?window_hours=24").json()
        assert len(d["forecast"]) >= 1


class TestTrace:
    def test_200(self, client, first_pid):
        assert client.get(f"/api/patients/{first_pid}/trace").status_code == 200

    def test_404(self, client):
        assert client.get("/api/patients/nope_xyz/trace").status_code == 404

    def test_schema(self, client, first_pid):
        d = client.get(f"/api/patients/{first_pid}/trace").json()
        for k in ["patient_id", "n_readings", "metrics", "trace"]:
            assert k in d

    def test_all_patients(self, client):
        for p in client.get("/api/patients").json()["patients"]:
            resp = client.get(f"/api/patients/{p['id']}/trace")
            assert resp.status_code == 200
            assert resp.json()["n_readings"] > 0


class TestExperiments:
    def test_200(self, client):
        assert client.get("/api/experiments").status_code == 200

    def test_phases_count(self, client):
        assert len(client.get("/api/experiments").json()["phases"]) >= 6

    def test_entry_schema(self, client):
        for p in client.get("/api/experiments").json()["phases"]:
            for k in ["id", "title", "plot_available", "report_available"]:
                assert k in p

    def test_plots_available(self, client):
        phases = client.get("/api/experiments").json()["phases"]
        assert sum(1 for p in phases if p["plot_available"]) >= 6

    def test_report_content(self, client):
        for p in client.get("/api/experiments").json()["phases"]:
            if p["report_available"]:
                resp = client.get(f"/api/experiments/{p['report']}/report")
                assert resp.status_code == 200
                assert len(resp.json()["content"]) > 10
                return
        pytest.skip("No report available")

    def test_image_served(self, client):
        for p in client.get("/api/experiments").json()["phases"]:
            if p["plot_available"]:
                resp = client.get(f"/api/experiments/{p['plot']}/image")
                assert resp.status_code == 200
                assert "image/png" in resp.headers["content-type"]
                return
        pytest.skip("No plot available")

    def test_non_png_rejected(self, client):
        assert client.get("/api/experiments/bad.exe/image").status_code == 400


class TestWhatIf:
    BASE = {
        "scenario_name": "Test", "duration_hours": 2,
        "meal_cho_g": 40.0, "meal_time_h": 0.5,
        "basal_insulin_mU_per_min": 15.0, "bolus_insulin_mU": 0.0, "seed": 42,
    }

    def test_200(self, client):
        assert client.post("/api/whatif", json=self.BASE).status_code == 200

    def test_schema(self, client):
        d = client.post("/api/whatif", json=self.BASE).json()
        for k in ["scenario_name", "data_origin", "disclaimer", "metrics", "trace"]:
            assert k in d

    def test_trace_nonempty(self, client):
        d = client.post("/api/whatif", json=self.BASE).json()
        assert len(d["trace"]) > 0
        for pt in d["trace"][:3]:
            assert "t_min" in pt and "glucose_mgdL" in pt
            assert 10 <= pt["glucose_mgdL"] <= 800

    def test_metrics_schema(self, client):
        m = client.post("/api/whatif", json=self.BASE).json()["metrics"]
        for k in ["tir_pct", "tbr_pct", "peak_glucose_mgdL", "min_glucose_mgdL"]:
            assert k in m

    def test_duration_affects_length(self, client):
        t2 = client.post("/api/whatif", json=dict(self.BASE, duration_hours=2)).json()["trace"]
        t4 = client.post("/api/whatif", json=dict(self.BASE, duration_hours=4)).json()["trace"]
        assert len(t4) > len(t2)

    def test_validation_error_too_large(self, client):
        assert client.post("/api/whatif", json=dict(self.BASE, meal_cho_g=9999)).status_code == 422

    def test_data_origin_synthetic(self, client):
        assert client.post("/api/whatif", json=self.BASE).json()["data_origin"] == "synthetic"

    def test_whatif_with_sleep_and_exercise(self, client):
        payload = dict(
            self.BASE,
            sleep_duration_hours=7.5,
            sleep_quality="Good",
            exercise_type="Running",
            exercise_duration_min=45,
            exercise_intensity="Moderate",
            exercise_start_time_h=1.0,
        )
        resp = client.post("/api/whatif", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert "sleep_scenario" in d
        assert d["sleep_scenario"]["duration_hours"] == 7.5
        assert d["sleep_scenario"]["modeled"] is False
        assert "exercise_scenario" in d
        assert d["exercise_scenario"]["type"] == "Running"
        assert d["exercise_scenario"]["modeled"] is False
        assert "interpretation" in d
        assert len(d["interpretation"]) > 10

    def test_whatif_patient_specific(self, client, first_pid):
        payload = dict(self.BASE, patient_id=first_pid)
        resp = client.post("/api/whatif", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert d["patient_id"] == first_pid
        assert "metrics" in d
        assert "initial_glucose_mgdL" in d["metrics"]
        assert "final_glucose_mgdL" in d["metrics"]


class TestAssistant:
    def test_assistant_query_basic(self, client, first_pid):
        payload = {
            "query": "Explain Time in Range",
            "patient_id": first_pid,
            "patient_name": "Synthetic Patient 000",
            "metrics": {"tir_pct": 82.5, "mean_glucose_mgdL": 115.0},
        }
        resp = client.post("/api/assistant/query", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert "answer" in d
        assert "70–180" in d["answer"] or "Time in Range" in d["answer"]
        assert d["disclaimer"]

    def test_assistant_query_safety(self, client):
        payload = {"query": "What insulin dose should I take?"}
        resp = client.post("/api/assistant/query", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert "SAFETY NOTICE" in d["answer"] or "medical device" in d["answer"].lower()


class TestPrivacy:
    def test_200(self, client):
        assert client.get("/api/privacy").status_code == 200

    def test_schema(self, client):
        d = client.get("/api/privacy").json()
        for k in ["summary", "data_stored", "data_not_stored", "network", "disclaimer"]:
            assert k in d

    def test_local_only(self, client):
        n = client.get("/api/privacy").json()["network"].lower()
        assert "local" in n or "no external" in n


class TestPatientParameters:
    def test_parameters_schema(self, client, first_pid):
        resp = client.get(f"/api/patients/{first_pid}/parameters")
        assert resp.status_code == 200
        d = resp.json()
        assert "parameters" in d
        assert "si_estimate" in d
        assert "sg_estimate" in d
        assert "model_context" in d
        assert "disclaimer" in d
        assert d["si_estimate"]["unit"] == "L / (mU · min)"
        assert d["sg_estimate"]["unit"] == "1 / min"

    def test_parameters_404(self, client):
        assert client.get("/api/patients/invalid_unknown_id/parameters").status_code == 404


class TestUserMeals:
    def test_meals_crud_lifecycle(self, client, first_pid):
        # 1. Get initial meals
        r0 = client.get(f"/api/patients/{first_pid}/meals")
        assert r0.status_code == 200
        initial_count = r0.json()["count"]

        # 2. Add a meal
        meal_payload = {
            "name": "Whole Wheat Toast & Eggs",
            "timestamp": "2026-10-09T08:00:00",
            "cho_g": 35.0,
            "category": "Breakfast",
            "notes": "Test nutrition record",
        }
        r1 = client.post(f"/api/patients/{first_pid}/meals", json=meal_payload)
        assert r1.status_code == 200
        created = r1.json()["meal"]
        assert created["name"] == "Whole Wheat Toast & Eggs"
        assert created["cho_g"] == 35.0
        assert created["is_user_logged"] is True
        meal_id = created["id"]

        # 3. Verify in get meals
        r2 = client.get(f"/api/patients/{first_pid}/meals")
        assert r2.json()["count"] == initial_count + 1

        # 4. Delete meal
        r3 = client.delete(f"/api/patients/{first_pid}/meals/{meal_id}")
        assert r3.status_code == 200

        # 5. Verify restored count
        r4 = client.get(f"/api/patients/{first_pid}/meals")
        assert r4.json()["count"] == initial_count

    def test_invalid_meal_rejected(self, client, first_pid):
        bad_payload = {"name": "", "timestamp": "invalid_time", "cho_g": -10.0}
        assert client.post(f"/api/patients/{first_pid}/meals", json=bad_payload).status_code == 422


class TestLiveSimulation:
    def test_simulation_execution(self, client, first_pid):
        payload = {
            "patient_id": first_pid,
            "duration_hours": 6.0,
            "step_minutes": 5.0,
            "include_user_meals": True,
        }
        resp = client.post("/api/simulation/run", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert d["data_origin"] == "simulated"
        assert "summary" in d
        assert "trace" in d
        assert len(d["trace"]) > 50
        s = d["summary"]
        assert s["tir_pct"] + s["tbr_pct"] + s["tar_pct"] == pytest.approx(100.0, abs=0.5)
        assert 20 <= s["min_glucose_mgdL"] <= s["max_glucose_mgdL"] <= 600

    def test_simulation_invalid_patient(self, client):
        payload = {"patient_id": "nonexistent_pid", "duration_hours": 6.0}
        assert client.post("/api/simulation/run", json=payload).status_code == 404


class TestDisclaimers:
    def test_all_endpoints_have_disclaimer(self, client, first_pid):
        endpoints = [
            "/api/health", "/api/patients", "/api/experiments", "/api/privacy",
            f"/api/overview/{first_pid}", f"/api/patients/{first_pid}/parameters",
            f"/api/patients/{first_pid}/meals", "/api/models/comparison",
            f"/api/patients/{first_pid}/ekf_estimate",
        ]
        for ep in endpoints:
            d = client.get(ep).json()
            assert "disclaimer" in d and len(d["disclaimer"]) > 5, f"Missing disclaimer: {ep}"


class TestHybridInference:
    def test_overview_hybrid_mode(self, client, first_pid):
        resp = client.get(f"/api/overview/{first_pid}?window_hours=24&model_mode=hybrid")
        assert resp.status_code == 200
        d = resp.json()
        assert d["model_mode"] in ["hybrid", "mechanistic"]
        assert "forecast" in d
        assert len(d["forecast"]) > 0
        for pt in d["forecast"]:
            assert "glucose_mgdL" in pt
            assert 20.0 <= pt["glucose_mgdL"] <= 600.0

    def test_whatif_hybrid_mode(self, client, first_pid):
        payload = {
            "scenario_name": "Hybrid Test Meal",
            "patient_id": first_pid,
            "duration_hours": 4.0,
            "meal_cho_g": 50.0,
            "meal_time_h": 1.0,
            "basal_insulin_mU_per_min": 15.0,
            "bolus_insulin_mU": 200.0,
            "model_mode": "hybrid",
        }
        resp = client.post("/api/whatif", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert d["model_mode"] in ["hybrid", "mechanistic"]
        assert "trace" in d
        assert len(d["trace"]) > 0


class TestEKFEstimation:
    def test_ekf_endpoint_success(self, client, first_pid):
        resp = client.get(f"/api/patients/{first_pid}/ekf_estimate?window_hours=12")
        assert resp.status_code == 200
        d = resp.json()
        assert d["status"] == "computed"
        assert d["data_origin"] == "estimated"
        assert "trace" in d
        assert len(d["trace"]) > 10
        first_pt = d["trace"][0]
        assert "cgm_observed_mgdL" in first_pt
        assert "glucose_est_mgdL" in first_pt
        assert "glucose_ci_lower_mgdL" in first_pt
        assert "glucose_ci_upper_mgdL" in first_pt
        assert "remote_insulin_action_est" in first_pt
        assert first_pt["glucose_ci_lower_mgdL"] <= first_pt["glucose_est_mgdL"] <= first_pt["glucose_ci_upper_mgdL"]

    def test_ekf_nonexistent_patient_404(self, client):
        assert client.get("/api/patients/nonexistent_xyz/ekf_estimate").status_code == 404


class TestModelComparison:
    def test_comparison_endpoint_structure(self, client):
        resp = client.get("/api/models/comparison")
        assert resp.status_code == 200
        d = resp.json()
        assert "records" in d
        assert len(d["records"]) >= 4
        for rec in d["records"]:
            assert "Horizon" in rec or "horizon" in rec
            assert "Model" in rec or "model" in rec
            assert "RMSE_mgdL" in rec or "rmse_mgdL" in rec
        assert "key_takeaways" in d
        assert len(d["key_takeaways"]) > 0

