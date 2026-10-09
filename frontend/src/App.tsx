import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
} from "recharts";
import axios from "axios";
import {
  Activity, HeartPulse, FlaskConical, History as HistoryIcon,
  Settings as SettingsIcon, RefreshCw, User, ShieldCheck, AlertTriangle,
  CheckCircle2, Info, Menu, X, Sliders, Utensils, TrendingUp, TrendingDown,
  ArrowRight, Play, Pause, RotateCcw, Plus, Trash2, ChevronRight, Send,
  Bot, Zap, Scale, Ruler, Clock, BarChart3, Target, Eye, Moon,
  ChevronLeft, ArrowLeftRight,
} from "lucide-react";

axios.defaults.timeout = 12000;

// ── Chart Visual Standards (Strictly standard across entire application) ──────
const CHART_COLORS = {
  cgmHistorical: "#047857",   // Solid Dark Green
  odeSimulation: "#0284C7",   // Solid Teal/Blue
  forecast: "#7C3AED",        // Dashed Purple
  lowThresh: "#DC2626",       // Dashed Red (70 mg/dL)
  highThresh: "#D97706",      // Dashed Amber (180 mg/dL)
  targetZone: "#ECFDF5",      // Shaded Light Green Band
};

// ── TypeScript Types ──────────────────────────────────────────────────────────
interface HealthStatus {
  status: string;
  api: string;
  dataset_ready: boolean;
  patient_count: number;
  mechanistic_model: string;
  experiment_plots: number;
}

interface ExperimentPhase {
  id: string;
  title: string;
  description: string;
  plot: string;
  report: string;
  data_origin: string;
  status: string;
  plot_available: boolean;
  report_available: boolean;
}

interface WhatIfResult {
  scenario_name: string;
  patient_id?: string | null;
  duration_hours?: number;
  metrics: {
    tir_pct: number;
    tbr_pct: number;
    tar_pct?: number;
    peak_glucose_mgdL: number;
    min_glucose_mgdL: number;
    mean_glucose_mgdL: number;
    initial_glucose_mgdL?: number;
    final_glucose_mgdL?: number;
    time_to_peak_h?: number;
    glucose_change_mgdL?: number;
  };
  trace: { t_min: number; glucose_mgdL: number }[];
  meal_time_h?: number;
  meal_cho_g?: number;
  bolus_insulin_mU?: number;
  sleep_scenario?: {
    duration_hours?: number | null;
    quality?: string;
    bedtime_h?: number | null;
    modeled: boolean;
    note: string;
  };
  exercise_scenario?: {
    type?: string;
    duration_min?: number;
    intensity?: string;
    start_time_h?: number | null;
    modeled: boolean;
    note: string;
  };
  interpretation?: string;
}

interface PatientTrace {
  patient_id: string;
  metrics: {
    mean_glucose_mgdL: number;
    std_glucose_mgdL: number;
    tir_pct: number;
    tbr_pct: number;
    tar_pct: number;
  };
  trace: { t: string; glucose_mgdL: number }[];
  n_readings: number;
}

interface PatientListItem {
  id: string;
  display_name: string;
  label: string;
  category: string;
  notes: string;
  data_origin: string;
  source: string;
  is_synthetic: boolean;
  mean_glucose_mgdL: number | null;
  tir_pct: number | null;
  n_readings: number;
  age?: number | null;
  weight_kg?: number | null;
  height_cm?: number | null;
  bmi?: number | null;
  sex?: string | null;
  baseline_glucose_mgdL?: number | null;
  created_at?: string;
}

interface OverviewData {
  patient_id: string;
  data_origin: string;
  disclaimer: string;
  units: string;
  window_hours: number;
  n_readings: number;
  current_glucose_mgdL: number;
  trend_arrow: string;
  metrics: {
    mean_glucose_mgdL: number;
    std_glucose_mgdL: number;
    tir_pct: number;
    tbr_pct: number;
    tar_pct: number;
  };
  cgm_trace: {
    t: string;
    glucose_mgdL: number;
    meal_cho_g: number;
    insulin_mU_per_min: number;
  }[];
  meal_events: {
    id?: string;
    t: string;
    cho_g: number;
    name?: string;
    category?: string;
    is_user_logged?: boolean;
  }[];
  forecast: {
    t: string;
    glucose_mgdL: number;
    is_forecast: boolean;
  }[];
  forecast_horizon_min: number;
  forecast_model: string;
  model_mode?: string;
  hybrid_available?: boolean;
  hybrid_error?: string | null;
}

interface EKFEstimateResponse {
  patient_id: string;
  filter_status: string;
  n_steps: number;
  step_minutes: number;
  mean_glucose_mgdL: number;
  final_state: {
    glucose_mgdL: number;
    insulin_action_per_min: number;
    plasma_insulin_mU_per_L: number;
  };
  estimated_sensitivity_Si: number;
  estimated_si?: number;
  window_hours?: number;
  trace?: any[];
  state_estimates: {
    t_min: number;
    measured_glucose_mgdL: number;
    estimated_glucose_mgdL: number;
    glucose_std_mgdL: number;
    ci_lower_mgdL: number;
    ci_upper_mgdL: number;
    insulin_action_X: number;
    plasma_insulin_I: number;
    innovation: number;
    remote_insulin_action_est?: number;
    glucose_est_mgdL?: number;
  }[];
  disclaimer: string;
}

interface ModelComparisonHorizonMetrics {
  rmse_mgdL: number;
  mae_mgdL: number;
  mard_pct: number;
  clarke_zone_a_pct: number;
  clarke_zone_b_pct: number;
  clarke_zone_a_plus_b_pct: number;
}

interface ModelComparisonModel {
  model_key: string;
  model_name: string;
  category: string;
  description: string;
  execution_status: string;
  horizons: Record<string, ModelComparisonHorizonMetrics>;
}

interface ModelComparisonResponse {
  title: string;
  evaluation_dataset: string;
  protocol: string;
  test_patient_id: string;
  n_test_samples: number;
  horizons: number[];
  models: ModelComparisonModel[];
  records?: any[];
  key_takeaways?: string[];
  summary_findings: string[];
  disclaimer: string;
}

interface PatientParameters {
  patient_id: string;
  calibration_status: string;
  parameters: {
    p1_Sg: number;
    p2: number;
    p3: number;
    n: number;
    Gb: number;
    Ib: number;
    Vg: number;
    Vi: number;
    rmse_calibrated: number;
  };
  si_estimate: {
    value: number;
    formatted: string;
    unit: string;
    symbol: string;
    name: string;
    plain_english: string;
  };
  sg_estimate: {
    value: number;
    formatted: string;
    unit: string;
    symbol: string;
    name: string;
    plain_english: string;
  };
  model_context: string;
}

interface UserMeal {
  id: string;
  patient_id: string;
  name: string;
  timestamp: string;
  cho_g: number;
  category: string;
  notes?: string;
  created_at?: string;
  is_user_logged: boolean;
}

interface LiveSimulationResponse {
  patient_id: string;
  data_origin: string;
  disclaimer: string;
  units: string;
  duration_hours: number;
  step_minutes: number;
  n_steps: number;
  parameters: Record<string, number>;
  summary: {
    initial_glucose_mgdL: number;
    final_glucose_mgdL: number;
    min_glucose_mgdL: number;
    max_glucose_mgdL: number;
    mean_glucose_mgdL: number;
    std_glucose_mgdL: number;
    tir_pct: number;
    tbr_pct: number;
    tar_pct: number;
    peak_glucose_mgdL: number;
    time_to_peak_h: number | null;
    time_to_return_h: number | null;
    total_meals_ingested: number;
    total_carbs_g: number;
  };
  trace: {
    step: number;
    t_min: number;
    timestamp: string;
    glucose_mgdL: number;
    is_simulated: boolean;
  }[];
  meals: {
    t_min: number;
    cho_g: number;
    name: string;
    is_user_logged: boolean;
  }[];
}

interface ChartPt {
  t: number;
  iso: string;
  glucose_mgdL?: number;
  sim_glucose_mgdL?: number;
  forecast_mgdL?: number;
}

// ── Constants & Helpers ───────────────────────────────────────────────────────
const PAGES = [
  { id: "overview",   label: "Overview",        icon: Activity },
  { id: "twin",       label: "My Digital Twin", icon: HeartPulse },
  { id: "whatif",     label: "What-If Lab",     icon: FlaskConical },
  { id: "comparison", label: "Model Benchmark", icon: BarChart3 },
  { id: "history",    label: "History",         icon: HistoryIcon },
  { id: "settings",   label: "Settings",        icon: SettingsIcon },
];

const LS_DEFAULT_WINDOW   = "t1d_default_window";
const LS_REFRESH_INTERVAL = "t1d_refresh_interval";
const LS_CHART_PREFS      = "t1d_chart_prefs";
const LS_LARGE_TEXT       = "t1d_large_text";
const DISCLAIMER = "Research prototype. Not a medical device. Not for clinical decisions.";

// ── Application Crash Protection (Error Boundary) ───────────────────────────
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an unhandled rendering error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ margin: "20px 0", borderLeft: "4px solid var(--status-danger-text)" }}>
          <div className="card-header">
            <h3 style={{ color: "var(--status-danger-text)", display: "flex", alignItems: "center", gap: 8 }}>
              <AlertTriangle size={18} />
              {this.props.fallbackTitle || "Research View Error"}
            </h3>
          </div>
          <div className="card-body">
            <p style={{ fontSize: 13.5, color: "var(--text-body)", marginBottom: 12 }}>
              An unexpected error occurred while rendering this research view. The platform sidebar and all other modules remain fully operational.
            </p>
            {this.state.error && (
              <div style={{ padding: "8px 12px", background: "var(--bg-secondary)", borderRadius: "var(--radius-xs)", fontSize: 12, color: "var(--text-secondary)", fontFamily: "monospace", marginBottom: 16 }}>
                {this.state.error.message || "Unknown error"}
              </div>
            )}
            <div style={{ display: "flex", gap: 10 }}>
              <button type="button" className="btn btn-primary" onClick={this.handleReset}>
                <RefreshCw size={13} />
                Try Again
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => window.location.reload()}
              >
                Reload Platform
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const TREND_INFO: Record<string, { text: string; icon: React.FC<any>; color: string }> = {
  "↑↑": { text: "Rising quickly", icon: TrendingUp,   color: "var(--status-danger-text)" },
  "↑":  { text: "Rising",          icon: TrendingUp,   color: "var(--status-warning-text)" },
  "→":  { text: "Steady / Stable", icon: ArrowRight,   color: "var(--emerald-600)" },
  "↓":  { text: "Falling",         icon: TrendingDown, color: "var(--status-warning-text)" },
  "↓↓": { text: "Falling quickly", icon: TrendingDown, color: "var(--status-danger-text)" },
};

const Spinner = () => <span className="loading-spinner" aria-label="Loading" />;

const Badge = ({ label, type = "neutral", icon: Icon }: { label: string; type?: string; icon?: React.FC<any> }) => (
  <span className={`badge badge-${type}`}>
    {Icon && <Icon size={12} />}
    <span>{label}</span>
  </span>
);

const tLabel = (t: number) => {
  const h = Math.floor(t / 60);
  const m = t % 60;
  return m === 0 ? `${h}h` : `${h}h${m}m`;
};

const avatarInitials = (name: string) => {
  const w = name.trim().split(/\s+/);
  return w.length >= 2 ? (w[0][0] + w[w.length - 1][0]).toUpperCase() : name.slice(0, 2).toUpperCase();
};

const calcBmi = (wkg: number | null | undefined, hcm: number | null | undefined): number | null =>
  wkg && hcm && hcm > 0 ? Math.round((wkg / Math.pow(hcm / 100, 2)) * 10) / 10 : null;

const bmiCategory = (b: number | null): string | null =>
  !b ? null : b < 18.5 ? "Underweight" : b < 25 ? "Normal" : b < 30 ? "Overweight" : "Obese";

const getGlucoseStatus = (g: number) =>
  g < 70  ? { type: "danger",  label: "Below Target (Low)",  cardCls: "danger" } :
  g > 180 ? { type: "warning", label: "Above Target (High)", cardCls: "warning" } :
             { type: "target",  label: "In Target Range",     cardCls: "target" };

// ── Standard Chart Legend Component ───────────────────────────────────────────
function ChartLegend({
  hasHistorical = false,
  hasSimulation = false,
  hasForecast = false,
  hasBaseline = false,
  showZones = true,
}: {
  hasHistorical?: boolean;
  hasSimulation?: boolean;
  hasForecast?: boolean;
  hasBaseline?: boolean;
  showZones?: boolean;
}) {
  return (
    <div className="chart-legend">
      {hasHistorical && (
        <div className="legend-item">
          <div className="legend-line-solid" style={{ background: CHART_COLORS.cgmHistorical }} />
          <span>━━ CGM Historical</span>
        </div>
      )}
      {hasSimulation && (
        <div className="legend-item">
          <div className="legend-line-solid" style={{ background: CHART_COLORS.odeSimulation }} />
          <span>━━ ODE Simulation</span>
        </div>
      )}
      {hasBaseline && (
        <div className="legend-item">
          <div className="legend-line-dashed" style={{ borderColor: "#64748B" }} />
          <span>╌╌ Previous / Baseline</span>
        </div>
      )}
      {hasForecast && (
        <div className="legend-item">
          <div className="legend-line-dashed" style={{ borderColor: CHART_COLORS.forecast }} />
          <span>╌╌ 30-minute Forecast</span>
        </div>
      )}
      {showZones && (
        <>
          <div className="legend-item">
            <div className="legend-line-dashed" style={{ borderColor: CHART_COLORS.lowThresh }} />
            <span>- - Low Threshold (70 mg/dL)</span>
          </div>
          <div className="legend-item">
            <div className="legend-line-dashed" style={{ borderColor: CHART_COLORS.highThresh }} />
            <span>- - High Threshold (180 mg/dL)</span>
          </div>
          <div className="legend-item">
            <div className="legend-band-box" />
            <span>Target Band (70–180 mg/dL)</span>
          </div>
        </>
      )}
    </div>
  );
}

// ── Compact Top Header (No Duplicate Navigation) ──────────────────────────────
function CompactHeader({
  page,
  sidebarCollapsed,
  setSidebarCollapsed,
  mobileOpen,
  setMobileOpen,
  health,
}: {
  page: string;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
  mobileOpen: boolean;
  setMobileOpen: (v: boolean) => void;
  health: HealthStatus | null;
}) {
  const curPage = PAGES.find(p => p.id === page) || PAGES[0];
  const PageIcon = curPage.icon;

  return (
    <header className={`compact-header ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <div className="header-left">
        {/* Desktop sidebar collapse toggle */}
        <button
          type="button"
          className="sidebar-toggle-btn"
          onClick={() => {
            if (window.innerWidth <= 768) {
              setMobileOpen(!mobileOpen);
            } else {
              setSidebarCollapsed(!sidebarCollapsed);
            }
          }}
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label="Toggle navigation sidebar"
        >
          {sidebarCollapsed ? <Menu size={17} /> : <ChevronLeft size={17} />}
        </button>

        <div className="header-breadcrumb">
          <PageIcon size={17} style={{ color: "var(--emerald-600)" }} />
          <span>{curPage.label}</span>
        </div>
      </div>

      <div className="header-right">
        {health ? (
          <Badge
            label={health.api === "online" ? "Backend Online" : "Backend Offline"}
            type={health.api === "online" ? "target" : "danger"}
            icon={health.api === "online" ? ShieldCheck : AlertTriangle}
          />
        ) : (
          <Badge label="Connecting..." type="neutral" icon={RefreshCw} />
        )}
        <Badge label="Local Execution Only" type="neutral" icon={ShieldCheck} />
        <div className="header-avatar" title="Research User Profile">
          <User size={16} />
        </div>
      </div>
    </header>
  );
}

// ── Permanent Left Sidebar ────────────────────────────────────────────────────
function Sidebar({
  page,
  setPage,
  collapsed,
  mobileOpen,
  setMobileOpen,
}: {
  page: string;
  setPage: (p: string) => void;
  collapsed: boolean;
  mobileOpen: boolean;
  setMobileOpen: (v: boolean) => void;
}) {
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`} aria-label="Main sidebar navigation">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <HeartPulse size={19} />
        </div>
        {!collapsed && (
          <div className="sidebar-brand-text">
            <h1>T1D Digital Twin</h1>
            <p>Diabetes Research Platform</p>
          </div>
        )}
      </div>

      <nav className="sidebar-nav">
        {!collapsed && <div className="sidebar-section-label">Navigation</div>}
        {PAGES.map(p => {
          const Icon = p.icon;
          const isActive = page === p.id;
          return (
            <button
              key={p.id}
              type="button"
              className={`sidebar-item ${isActive ? "active" : ""}`}
              onClick={() => {
                setPage(p.id);
                setMobileOpen(false);
              }}
              title={collapsed ? p.label : undefined}
            >
              <Icon size={18} style={{ flexShrink: 0 }} />
              {!collapsed && <span className="sidebar-item-text">{p.label}</span>}
            </button>
          );
        })}
      </nav>

      {!collapsed && (
        <div className="sidebar-footer">
          <div className="disclaimer-box">
            <p style={{ fontWeight: 600, color: "var(--text-main)", marginBottom: 2 }}>Research Prototype</p>
            <p>Not a medical device. Not for clinical decisions. All data synthetic.</p>
          </div>
        </div>
      )}
    </aside>
  );
}

// ── Create Patient Modal ──────────────────────────────────────────────────────
function CreatePatientModal({ onClose, onCreated }: {
  onClose: () => void; onCreated: (p: PatientListItem) => void;
}) {
  const [form, setForm] = useState({ display_name: "", age: "", weight_kg: "", height_cm: "", sex: "", notes: "" });
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.display_name.trim()) { setErr("Name is required."); return; }
    setSaving(true); setErr("");
    try {
      const payload: any = { display_name: form.display_name.trim() };
      if (form.age)       payload.age       = parseInt(form.age, 10);
      if (form.weight_kg) payload.weight_kg = parseFloat(form.weight_kg);
      if (form.height_cm) payload.height_cm = parseFloat(form.height_cm);
      if (form.sex)       payload.sex       = form.sex;
      if (form.notes)     payload.notes     = form.notes;
      const r = await axios.post("/api/patients", payload);
      onCreated({ ...r.data, label: r.data.display_name, source: "user", tir_pct: null, mean_glucose_mgdL: null, n_readings: 0 });
    } catch (ex: any) {
      setErr(ex.response?.data?.detail || "Failed to create patient.");
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-panel">
        <div className="modal-header">
          <h3><User size={16} />Create Virtual Patient Profile</h3>
          <button type="button" className="btn btn-ghost" style={{ padding: "4px 8px" }} onClick={onClose}>
            <X size={16} />
          </button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {err && (
              <div className="error-box mb-3">
                <AlertTriangle size={15} />
                <span>{err}</span>
              </div>
            )}
            <div className="form-group">
              <label className="form-label">Patient Display Name *</label>
              <input
                className="form-input"
                value={form.display_name}
                onChange={e => set("display_name", e.target.value)}
                placeholder="e.g. Synthetic Patient 005"
                maxLength={100}
                required
              />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Age (years)</label>
                <input
                  className="form-input"
                  type="number"
                  min={1}
                  max={120}
                  value={form.age}
                  onChange={e => set("age", e.target.value)}
                  placeholder="e.g. 28"
                />
              </div>
              <div className="form-group">
                <label className="form-label">Biological Sex</label>
                <select className="form-select" value={form.sex} onChange={e => set("sex", e.target.value)}>
                  <option value="">Select...</option>
                  <option>Male</option>
                  <option>Female</option>
                  <option>Non-binary</option>
                  <option>Prefer not to say</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Weight (kg)</label>
                <input
                  className="form-input"
                  type="number"
                  step="0.1"
                  min={1}
                  max={500}
                  value={form.weight_kg}
                  onChange={e => set("weight_kg", e.target.value)}
                  placeholder="70.0"
                />
              </div>
              <div className="form-group">
                <label className="form-label">Height (cm)</label>
                <input
                  className="form-input"
                  type="number"
                  step="0.1"
                  min={30}
                  max={300}
                  value={form.height_cm}
                  onChange={e => set("height_cm", e.target.value)}
                  placeholder="175.0"
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Research Notes (optional)</label>
              <textarea
                className="form-textarea"
                rows={2}
                value={form.notes}
                onChange={e => set("notes", e.target.value)}
                maxLength={500}
                placeholder="e.g. Virtual cohort benchmark profile."
              />
            </div>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? <><Spinner /> Creating...</> : <><Plus size={14} /> Create Profile</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── Virtual Patient List Panel ────────────────────────────────────────────────
function PatientListPanel({ patients, selectedId, onSelect, onRefresh, loading }: {
  patients: PatientListItem[]; selectedId: string;
  onSelect: (id: string) => void; onRefresh: () => void; loading: boolean;
}) {
  const [showCreate, setShowCreate] = useState(false);
  const [localList, setLocalList] = useState(patients);
  useEffect(() => setLocalList(patients), [patients]);

  const handleCreated = (p: PatientListItem) => {
    setLocalList(prev => [p, ...prev]);
    onSelect(p.id);
    setShowCreate(false);
    onRefresh();
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Delete this patient and all associated meal records?")) return;
    try {
      await axios.delete(`/api/patients/${id}`);
      setLocalList(prev => prev.filter(p => p.id !== id));
      onRefresh();
    } catch {
      alert("Failed to delete patient.");
    }
  };

  const list = localList.length > 0 ? localList : patients;

  return (
    <>
      <div className="card">
        <div className="card-header">
          <h3><User size={15} />Virtual Patients</h3>
          <button
            type="button"
            className="btn btn-ghost"
            style={{ padding: "4px 8px" }}
            onClick={onRefresh}
            title="Refresh patient list"
          >
            <RefreshCw size={13} />
          </button>
        </div>
        <div className="patient-list">
          {loading ? (
            <div style={{ padding: 24, textAlign: "center" }}><Spinner /></div>
          ) : list.length === 0 ? (
            <div className="empty-state"><p>No virtual patients found.</p></div>
          ) : (
            list.map(p => (
              <div
                key={p.id}
                className={`patient-list-item ${selectedId === p.id ? "selected" : ""}`}
                onClick={() => onSelect(p.id)}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === "Enter" && onSelect(p.id)}
              >
                <div className={`patient-avatar ${p.is_synthetic ? "synthetic" : "user"}`}>
                  {avatarInitials(p.display_name)}
                </div>
                <div className="patient-meta">
                  <div className="patient-name">{p.display_name}</div>
                  <div className="patient-sub">
                    {p.tir_pct != null ? `TIR ${p.tir_pct.toFixed(1)}%` : "Custom Profile"} · {p.category}
                  </div>
                </div>
                {!p.is_synthetic && (
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "3px 6px", opacity: 0.6 }}
                    onClick={e => handleDelete(p.id, e)}
                    title="Delete patient"
                  >
                    <Trash2 size={13} style={{ color: "var(--status-danger-text)" }} />
                  </button>
                )}
                <ChevronRight size={14} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
              </div>
            ))
          )}
        </div>
        <button type="button" className="create-patient-btn" onClick={() => setShowCreate(true)}>
          <Plus size={14} /> Add New Patient
        </button>
      </div>
      {showCreate && <CreatePatientModal onClose={() => setShowCreate(false)} onCreated={handleCreated} />}
    </>
  );
}

// ── Patient Profile Card ──────────────────────────────────────────────────────
function ProfileCard({ profile, params }: { profile: PatientListItem | null; params: PatientParameters | null; }) {
  if (!profile) {
    return (
      <div className="card">
        <div className="card-body empty-state">
          <p>Select a patient to view physiological profile.</p>
        </div>
      </div>
    );
  }

  const bmi = calcBmi(profile.weight_kg, profile.height_cm);
  const bmiCat = bmiCategory(bmi);

  return (
    <div className="card">
      <div className="card-header">
        <h3><User size={15} />Patient Profile</h3>
        <Badge
          label={profile.is_synthetic ? "Synthetic Benchmark" : "Custom Profile"}
          type={profile.is_synthetic ? "neutral" : "teal"}
        />
      </div>
      <div className="profile-card">
        <div className="profile-identity">
          <div className={`profile-big-avatar ${profile.is_synthetic ? "synthetic" : "user"}`}>
            {avatarInitials(profile.display_name)}
          </div>
          <div className="profile-id-text">
            <h4>{profile.display_name}</h4>
            <p>{profile.category} · {profile.id}</p>
            {profile.notes && (
              <p style={{ marginTop: 4, fontSize: 12, color: "var(--text-secondary)", fontStyle: "italic" }}>
                {profile.notes}
              </p>
            )}
          </div>
        </div>

        {[
          { icon: User,     label: "Age",    val: profile.age       ? `${profile.age} yrs`      : "—" },
          { icon: Scale,    label: "Weight", val: profile.weight_kg ? `${profile.weight_kg} kg` : "—" },
          { icon: Ruler,    label: "Height", val: profile.height_cm ? `${profile.height_cm} cm` : "—" },
          { icon: Activity, label: "BMI",    val: bmi ? `${bmi}` : "—", sub: bmiCat ?? "" },
        ].map((s: any) => (
          <div key={s.label} className="profile-stat">
            <div className="stat-icon"><s.icon size={12} /> {s.label}</div>
            <div className="stat-val">{s.val} {s.sub && <span className="stat-unit">({s.sub})</span>}</div>
          </div>
        ))}

        {params && (
          <div style={{ gridColumn: "1 / -1", marginTop: 4 }}>
            <div style={{ fontSize: 11.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-secondary)", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <Zap size={13} style={{ color: "var(--emerald-600)" }} />
              Bergman Model Parameters
            </div>
            <table className="baseline-table">
              <tbody>
                <tr>
                  <td>Fasting Glucose (G<sub>b</sub>)</td>
                  <td>{params.parameters.Gb.toFixed(1)} mg/dL</td>
                </tr>
                <tr>
                  <td>Baseline Insulin (I<sub>b</sub>)</td>
                  <td>{params.parameters.Ib.toFixed(1)} mU/L</td>
                </tr>
                <tr>
                  <td>Insulin Sensitivity (S<sub>I</sub>)</td>
                  <td>{params.si_estimate.formatted} <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{params.si_estimate.unit}</span></td>
                </tr>
                <tr>
                  <td>Glucose Effectiveness (S<sub>G</sub>)</td>
                  <td>{params.sg_estimate.formatted} <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{params.sg_estimate.unit}</span></td>
                </tr>
                <tr>
                  <td>Calibration Fit RMSE</td>
                  <td>{params.parameters.rmse_calibrated.toFixed(2)} mg/dL</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 8, lineHeight: 1.4 }}>
              * Calibrated from benchmark in silico dataset. Not a direct clinical measurement.
            </div>
          </div>
        )}

        {!params && !profile.is_synthetic && (
          <div style={{ gridColumn: "1 / -1", marginTop: 4 }}>
            <div style={{ fontSize: 11.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-secondary)", marginBottom: 8 }}>
              Baseline Configuration
            </div>
            <table className="baseline-table">
              <tbody>
                <tr>
                  <td>Baseline Glucose</td>
                  <td>{profile.baseline_glucose_mgdL ? `${profile.baseline_glucose_mgdL} mg/dL` : "100.0 mg/dL (Default)"}</td>
                </tr>
                <tr>
                  <td>Simulation ODE Model</td>
                  <td>Bergman Minimal Model</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Simulation Controls Card ──────────────────────────────────────────────────
function SimulationControls({ simState, simDuration, setSimDuration, onStart, onPause, onResume, onReset, loading }: {
  simState: "idle" | "running" | "paused" | "done";
  simDuration: number; setSimDuration: (v: number) => void;
  onStart: () => void; onPause: () => void; onResume: () => void; onReset: () => void;
  loading: boolean;
}) {
  return (
    <div className="card">
      <div className="card-header">
        <h3><Sliders size={15} />Physiological Simulation</h3>
        <Badge
          label={simState === "idle" ? "Idle" : simState === "running" ? "Simulating" : simState === "paused" ? "Paused" : "Finished"}
          type={simState === "running" ? "target" : simState === "done" ? "teal" : "neutral"}
        />
      </div>
      <div className="card-body">
        <div className="form-group" style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <label className="form-label" style={{ margin: 0 }}>Simulation Horizon</label>
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--emerald-700)" }}>{simDuration} Hours</span>
          </div>
          <input
            type="range"
            min={1}
            max={24}
            step={1}
            value={simDuration}
            onChange={e => setSimDuration(Number(e.target.value))}
            disabled={simState === "running"}
            style={{ width: "100%", accentColor: "var(--emerald-600)" }}
          />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {simState === "idle" && (
            <button className="btn btn-primary w-full" onClick={onStart} disabled={loading}>
              {loading ? <><Spinner /> Running ODE...</> : <><Play size={14} /> Run Simulation</>}
            </button>
          )}
          {simState === "running" && (
            <>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={onPause}>
                <Pause size={14} /> Pause
              </button>
              <button className="btn btn-ghost" onClick={onReset} title="Reset simulation">
                <RotateCcw size={14} />
              </button>
            </>
          )}
          {simState === "paused" && (
            <>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={onResume}>
                <Play size={14} /> Resume
              </button>
              <button className="btn btn-ghost" onClick={onReset} title="Reset simulation">
                <RotateCcw size={14} />
              </button>
            </>
          )}
          {simState === "done" && (
            <button className="btn btn-secondary w-full" onClick={onReset}>
              <RotateCcw size={14} /> Reset Simulation
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Interactive CGM Chart with Strictly Standardized Visual Encoding ───────────
function CGMChart({
  chartData,
  simResponse,
  showForecast,
  showZones,
  title = "CGM Glucose Monitor",
}: {
  chartData: ChartPt[];
  simResponse: LiveSimulationResponse | null;
  showForecast: boolean;
  showZones: boolean;
  title?: string;
}) {
  const hasHistorical = chartData.some(d => d.glucose_mgdL !== undefined);
  const hasSimulation = chartData.some(d => d.sim_glucose_mgdL !== undefined);
  const hasForecastData = showForecast && chartData.some(d => d.forecast_mgdL !== undefined);

  return (
    <div className="card">
      <div className="card-header">
        <h3><Activity size={15} />{title}</h3>
        <Badge
          label={simResponse ? "ODE Simulation Active" : "Historical Sensor"}
          type={simResponse ? "target" : "neutral"}
          icon={ShieldCheck}
        />
      </div>
      <div className="card-body">
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 14, left: -14, bottom: 0 }}>
              <defs>
                <linearGradient id="cgmGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={CHART_COLORS.cgmHistorical} stopOpacity={0.18} />
                  <stop offset="95%" stopColor={CHART_COLORS.cgmHistorical} stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={CHART_COLORS.odeSimulation} stopOpacity={0.22} />
                  <stop offset="95%" stopColor={CHART_COLORS.odeSimulation} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 4" stroke="#E2E8F0" vertical={false} />
              <XAxis
                dataKey="t"
                stroke="#94A3B8"
                tick={{ fontSize: 11, fill: "#64748B" }}
                tickFormatter={tLabel}
                interval={Math.max(1, Math.floor(chartData.length / 8))}
              />
              <YAxis
                domain={[40, 320]}
                stroke="#94A3B8"
                tick={{ fontSize: 11, fill: "#64748B" }}
                tickCount={8}
              />
              <Tooltip
                contentStyle={{
                  background: "#FFFFFF",
                  border: "1px solid #E2E8F0",
                  borderRadius: 10,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                  fontSize: 12.5,
                }}
                labelStyle={{ color: "#0F172A", fontWeight: 700 }}
                formatter={(v: any, name: any) => [
                  `${Number(v ?? 0).toFixed(1)} mg/dL`,
                  name === "glucose_mgdL" ? "CGM Sensor" : name === "sim_glucose_mgdL" ? "ODE Simulation" : "30-min Forecast",
                ]}
                labelFormatter={(t: any) => `Time = ${tLabel(Number(t))}`}
              />
              {showZones && (
                <ReferenceArea y1={70} y2={180} fill={CHART_COLORS.targetZone} fillOpacity={0.75} stroke="none" />
              )}
              {showZones && (
                <ReferenceLine
                  y={70}
                  stroke={CHART_COLORS.lowThresh}
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{ value: "Low 70", fill: CHART_COLORS.lowThresh, fontSize: 10, position: "insideBottomLeft" }}
                />
              )}
              {showZones && (
                <ReferenceLine
                  y={180}
                  stroke={CHART_COLORS.highThresh}
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{ value: "High 180", fill: CHART_COLORS.highThresh, fontSize: 10, position: "insideTopLeft" }}
                />
              )}
              {/* CGM Historical: Solid Dark Green */}
              <Area
                type="monotone"
                dataKey="glucose_mgdL"
                stroke={CHART_COLORS.cgmHistorical}
                strokeWidth={2.5}
                fill="url(#cgmGrad)"
                dot={false}
                activeDot={{ r: 4 }}
                name="glucose_mgdL"
              />
              {/* ODE Simulation: Solid Teal/Blue */}
              <Area
                type="monotone"
                dataKey="sim_glucose_mgdL"
                stroke={CHART_COLORS.odeSimulation}
                strokeWidth={2.5}
                fill="url(#simGrad)"
                dot={false}
                activeDot={{ r: 4 }}
                name="sim_glucose_mgdL"
              />
              {/* 30-min Forecast: Dashed Purple */}
              {showForecast && (
                <Area
                  type="monotone"
                  dataKey="forecast_mgdL"
                  stroke={CHART_COLORS.forecast}
                  strokeWidth={2}
                  strokeDasharray="5 4"
                  fill="none"
                  dot={false}
                  activeDot={{ r: 4 }}
                  name="forecast_mgdL"
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <ChartLegend
          hasHistorical={hasHistorical}
          hasSimulation={hasSimulation}
          hasForecast={hasForecastData}
          showZones={showZones}
        />
      </div>
    </div>
  );
}

// ── 5 Key Metrics Row ─────────────────────────────────────────────────────────
function KeyMetricsRow({ data, simResponse }: { data: OverviewData | null; simResponse?: LiveSimulationResponse | null; }) {
  const tir  = simResponse ? simResponse.summary.tir_pct          : data?.metrics.tir_pct;
  const mean = simResponse ? simResponse.summary.mean_glucose_mgdL : data?.metrics.mean_glucose_mgdL;
  const tbr  = simResponse ? simResponse.summary.tbr_pct           : data?.metrics.tbr_pct;
  const tar  = simResponse ? simResponse.summary.tar_pct           : data?.metrics.tar_pct;
  const sd   = simResponse ? simResponse.summary.std_glucose_mgdL  : data?.metrics.std_glucose_mgdL;

  const cards = [
    {
      label: "Time in Range (TIR)",
      val: tir != null ? `${tir.toFixed(1)}%` : "—",
      sub: "Target: ≥70%",
      Icon: Target,
      cls: tir != null && tir >= 70 ? "success" : "warning",
    },
    {
      label: "Time Below Range",
      val: tbr != null ? `${tbr.toFixed(1)}%` : "—",
      sub: "Target: <4%",
      Icon: AlertTriangle,
      cls: tbr != null && tbr > 4 ? "danger" : "success",
    },
    {
      label: "Time Above Range",
      val: tar != null ? `${tar.toFixed(1)}%` : "—",
      sub: "Target: <25%",
      Icon: TrendingUp,
      cls: tar != null && tar > 25 ? "warning" : "success",
    },
    {
      label: "Mean Glucose",
      val: mean != null ? `${mean.toFixed(1)}` : "—",
      sub: "mg/dL",
      Icon: BarChart3,
      cls: "neutral",
    },
    {
      label: "Variability (SD)",
      val: sd != null ? `${sd.toFixed(1)}` : "—",
      sub: "mg/dL",
      Icon: Activity,
      cls: "neutral",
    },
  ];

  return (
    <div className="metrics-row">
      {cards.map(t => (
        <div key={t.label} className="metric-card">
          <div className={`metric-icon-wrap ${t.cls}`}>
            <t.Icon size={18} />
          </div>
          <div className="metric-info">
            <div className="metric-label">{t.label}</div>
            <div className="metric-value">{t.val}</div>
            <div className="metric-unit">{t.sub}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Health Insights Panel ─────────────────────────────────────────────────────
function HealthInsights({ data, simResponse }: { data: OverviewData | null; simResponse?: LiveSimulationResponse | null; }) {
  const tir  = simResponse ? simResponse.summary.tir_pct  : data?.metrics.tir_pct;
  const tbr  = simResponse ? simResponse.summary.tbr_pct  : data?.metrics.tbr_pct;
  const tar  = simResponse ? simResponse.summary.tar_pct  : data?.metrics.tar_pct;
  const peak = simResponse?.summary.peak_glucose_mgdL;
  const ttp  = simResponse?.summary.time_to_peak_h;
  const ttr  = simResponse?.summary.time_to_return_h;

  const insights: { Icon: React.FC<any>; color: string; text: string }[] = [];

  if (tir != null) {
    insights.push({
      Icon: tir >= 70 ? CheckCircle2 : AlertTriangle,
      color: tir >= 70 ? "var(--emerald-600)" : "var(--status-warning-text)",
      text: `Time in Range is ${tir.toFixed(1)}% — ${tir >= 70 ? "meeting" : "below"} the ≥70% consensus research goal.`,
    });
  }
  if (tbr != null && tbr > 4) {
    insights.push({
      Icon: AlertTriangle,
      color: "var(--status-danger-text)",
      text: `Time below range is ${tbr.toFixed(1)}%, exceeding the <4% hypoglycemia research threshold.`,
    });
  }
  if (tar != null && tar > 25) {
    insights.push({
      Icon: Info,
      color: "var(--status-warning-text)",
      text: `Time above range is ${tar.toFixed(1)}%. Target guideline is <25%.`,
    });
  }
  if (peak != null && ttp != null) {
    insights.push({
      Icon: TrendingUp,
      color: "var(--text-body)",
      text: `Simulation estimated peak glucose of ${peak.toFixed(1)} mg/dL at ${(ttp * 60).toFixed(0)} min after start.`,
    });
  }
  if (ttr != null) {
    insights.push({
      Icon: Clock,
      color: "var(--text-body)",
      text: `Glucose returned near baseline after ${(ttr * 60).toFixed(0)} min under minimal model ODE dynamics.`,
    });
  }
  if (simResponse) {
    insights.push({
      Icon: ShieldCheck,
      color: "var(--emerald-600)",
      text: `Simulation integrated ${simResponse.meals.length} meal event(s) totaling ${simResponse.summary.total_carbs_g}g carbohydrates.`,
    });
  }
  if (insights.length === 0) {
    insights.push({
      Icon: Info,
      color: "var(--text-muted)",
      text: "Load patient telemetry or run an ODE simulation to generate real-time physiological insights.",
    });
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3><Eye size={15} />Physiological Insights</h3>
      </div>
      <div className="card-body" style={{ padding: "10px 18px" }}>
        {insights.map((ins, i) => (
          <div key={i} className="insight-item">
            <ins.Icon size={15} style={{ color: ins.color, flexShrink: 0, marginTop: 2 }} />
            <span style={{ fontSize: 13, color: "var(--text-main)" }}>{ins.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Context-Aware Research Assistant ──────────────────────────────────────────
function HealthAssistant({
  patientId,
  patientName,
  metrics,
  simulationSummary,
  scenarioInfo,
  sleepInfo,
  exerciseInfo,
}: {
  patientId?: string;
  patientName: string;
  metrics?: any;
  simulationSummary?: any;
  scenarioInfo?: any;
  sleepInfo?: any;
  exerciseInfo?: any;
}) {
  const [messages, setMessages] = useState<{ role: "assistant" | "user"; text: string }[]>([
    {
      role: "assistant",
      text: `Hello! I am your Research Assistant. Ask me questions about ${patientName}'s glucose telemetry, the Bergman ODE Minimal Model, or simulation parameters.`,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, loading]);

  const QUICK_PROMPTS = [
    "Explain this glucose graph",
    "Explain Time in Range",
    "Compare historical glucose with the simulation",
    "Summarize this simulation",
    "Explain the effect of the selected meal",
    "Explain the configured exercise scenario",
    "Explain how the sleep input is represented in the model",
  ];

  const ask = async (q: string) => {
    if (!q.trim() || loading) return;
    const userQuery = q.trim();
    setMessages(prev => [...prev, { role: "user", text: userQuery }]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("/api/assistant/query", {
        query: userQuery,
        patient_id: patientId,
        patient_name: patientName,
        metrics: metrics,
        simulation_summary: simulationSummary,
        scenario_info: scenarioInfo,
        sleep_info: sleepInfo,
        exercise_info: exerciseInfo,
      });
      setMessages(prev => [...prev, { role: "assistant", text: res.data.answer }]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          text: "Unable to reach research assistant service. Ensure the FastAPI backend is online at port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3><Bot size={15} />Research Assistant</h3>
        <Badge label="Context-Aware Engine" type="neutral" />
      </div>
      <div className="assistant-body" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className="assistant-msg" style={m.role === "user" ? { flexDirection: "row-reverse" } : {}}>
            {m.role === "assistant" && (
              <div className="assistant-avatar">
                <Bot size={14} />
              </div>
            )}
            <div className={`assistant-bubble ${m.role === "user" ? "user-msg" : ""}`}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="assistant-msg">
            <div className="assistant-avatar">
              <Bot size={14} />
            </div>
            <div className="assistant-bubble" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Spinner />
              <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Synthesizing model context...</span>
            </div>
          </div>
        )}
      </div>
      <div className="assistant-quick-btns">
        {QUICK_PROMPTS.map(q => (
          <button key={q} className="assistant-quick-btn" onClick={() => ask(q)} disabled={loading}>
            {q}
          </button>
        ))}
      </div>
      <div className="assistant-input-row">
        <input
          className="assistant-input"
          placeholder={`Ask about ${patientName}'s metrics, simulation, or Bergman ODE...`}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && ask(input)}
          disabled={loading}
        />
        <button
          className="assistant-send-btn"
          onClick={() => ask(input)}
          title="Send query"
          disabled={loading || !input.trim()}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}

// ── Meal Logger Component ─────────────────────────────────────────────────────
function MealLogger({ patientId, patientName }: { patientId: string; patientName: string; }) {
  const [meals, setMeals] = useState<UserMeal[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [form, setForm] = useState({
    name: "",
    timestamp: new Date().toISOString().slice(0, 16),
    cho_g: "",
    category: "Lunch",
    notes: "",
  });

  const fetchMeals = useCallback(() => {
    if (!patientId) return;
    setLoading(true);
    axios.get(`/api/patients/${patientId}/meals`)
      .then(r => setMeals(r.data.meals || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [patientId]);

  useEffect(() => { fetchMeals(); }, [fetchMeals]);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim() || !form.cho_g) {
      setErr("Meal name and carbohydrate amount are required.");
      return;
    }
    setSaving(true);
    setErr("");
    setMsg("");
    try {
      const r = await axios.post(`/api/patients/${patientId}/meals`, {
        name: form.name.trim(),
        timestamp: new Date(form.timestamp).toISOString(),
        cho_g: parseFloat(form.cho_g),
        category: form.category,
        notes: form.notes,
      });
      setMeals(prev => [...prev, r.data.meal]);
      setMsg("Meal successfully logged.");
      setForm(f => ({ ...f, name: "", cho_g: "", notes: "" }));
    } catch (ex: any) {
      setErr(ex.response?.data?.detail || "Failed to log meal.");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await axios.delete(`/api/patients/${patientId}/meals/${id}`);
      setMeals(prev => prev.filter(m => m.id !== id));
    } catch {
      alert("Failed to delete meal.");
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3><Utensils size={15} />Nutrition &amp; Meal Logger — {patientName}</h3>
        <Badge label={`${meals.length} Logged`} type="neutral" />
      </div>
      <div className="card-body">
        {msg && (
          <div style={{ marginBottom: 12, padding: "8px 12px", background: "var(--emerald-50)", border: "1px solid var(--emerald-200)", borderRadius: 8, fontSize: 13, color: "var(--emerald-800)", display: "flex", alignItems: "center", gap: 6 }}>
            <CheckCircle2 size={14} />{msg}
          </div>
        )}
        {err && (
          <div className="error-box mb-3">
            <AlertTriangle size={14} />{err}
          </div>
        )}
        <form onSubmit={handleAdd} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Meal Name *</label>
            <input
              className="form-input"
              value={form.name}
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Oatmeal with Fruit"
              required
            />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Carbs (g) *</label>
            <input
              className="form-input"
              type="number"
              step="1"
              min={0}
              max={300}
              value={form.cho_g}
              onChange={e => setForm(f => ({ ...f, cho_g: e.target.value }))}
              placeholder="45"
              required
            />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Date &amp; Time</label>
            <input
              className="form-input"
              type="datetime-local"
              value={form.timestamp}
              onChange={e => setForm(f => ({ ...f, timestamp: e.target.value }))}
            />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Meal Category</label>
            <select
              className="form-select"
              value={form.category}
              onChange={e => setForm(f => ({ ...f, category: e.target.value }))}
            >
              {["Breakfast", "Lunch", "Dinner", "Snack", "Correction"].map(c => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="form-group" style={{ margin: 0, gridColumn: "1 / -1" }}>
            <label className="form-label">Notes (optional)</label>
            <input
              className="form-input"
              value={form.notes}
              onChange={e => setForm(f => ({ ...f, notes: e.target.value }))}
              placeholder="e.g. Post-exercise breakfast"
            />
          </div>
          <div style={{ gridColumn: "1 / -1" }}>
            <button type="submit" className="btn btn-primary w-full" disabled={saving}>
              {saving ? <><Spinner /> Saving...</> : <><Plus size={14} /> Log Meal Event</>}
            </button>
          </div>
        </form>

        {loading ? (
          <div style={{ textAlign: "center", padding: 12 }}><Spinner /></div>
        ) : (
          <div style={{ maxHeight: 200, overflowY: "auto" }}>
            {meals.length === 0 ? (
              <div className="empty-state" style={{ padding: 16 }}>
                <p>No user meals recorded for this patient.</p>
              </div>
            ) : (
              meals.map(m => (
                <div
                  key={m.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "8px 0",
                    borderBottom: "1px solid var(--border-color)",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-main)" }}>{m.name}</div>
                    <div style={{ fontSize: 11.5, color: "var(--text-secondary)" }}>
                      {m.category} · {m.cho_g}g carbs · {new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "4px 6px" }}
                    onClick={() => handleDelete(m.id)}
                    title="Remove meal"
                  >
                    <Trash2 size={13} style={{ color: "var(--status-danger-text)" }} />
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Overview Page ─────────────────────────────────────────────────────────────
function OverviewPage({
  selectedId,
  setSelectedId,
  profileNames,
  setProfileName,
  defaultWindowHours,
  refreshInterval,
  chartPrefs,
}: {
  selectedId: string;
  setSelectedId: (id: string) => void;
  profileNames: Record<string, string>;
  setProfileName: (id: string, n: string) => void;
  defaultWindowHours: number;
  refreshInterval: number;
  chartPrefs: { showArea: boolean; showForecast: boolean; showZones: boolean };
}) {
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [windowHours, setWindowHours] = useState(defaultWindowHours);
  const [modelMode, setModelMode] = useState<"mechanistic" | "hybrid">("mechanistic");
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState("");
  const [dataError, setDataError] = useState("");
  const [nameInput, setNameInput] = useState("");
  const [nameError, setNameError] = useState("");

  useEffect(() => setWindowHours(defaultWindowHours), [defaultWindowHours]);

  useEffect(() => {
    setListError("");
    axios.get("/api/patients")
      .then(r => {
        const pts: PatientListItem[] = r.data.patients;
        setPatients(pts);
        if (pts.length > 0 && !selectedId) setSelectedId(pts[0].id);
      })
      .catch(err => setListError(err.code === "ECONNABORTED" ? "Connection timed out." : "Backend offline."))
      .finally(() => setListLoading(false));
  }, [selectedId, setSelectedId]);

  useEffect(() => {
    if (selectedId) {
      const found = patients.find(p => p.id === selectedId);
      setNameInput(profileNames[selectedId] || found?.display_name || selectedId);
      setNameError("");
    }
  }, [selectedId, profileNames, patients]);

  const fetchOverview = useCallback((isManual = false) => {
    if (!selectedId) return;
    if (isManual) setRefreshing(true); else setLoading(true);
    setDataError("");
    axios.get(`/api/overview/${selectedId}`, { params: { window_hours: windowHours, model_mode: modelMode } })
      .then(r => {
        setData(r.data);
        setLastRefreshed(new Date());
      })
      .catch(() => setDataError("Unable to retrieve CGM telemetry. Check backend service."))
      .finally(() => { setLoading(false); setRefreshing(false); });
  }, [selectedId, windowHours, modelMode]);

  useEffect(() => { fetchOverview(false); }, [fetchOverview]);

  useEffect(() => {
    if (!refreshInterval || refreshInterval <= 0) return;
    const t = setInterval(() => fetchOverview(true), refreshInterval * 1000);
    return () => clearInterval(t);
  }, [refreshInterval, fetchOverview]);

  const chartData = useMemo(() => {
    if (!data) return [];
    const h = data.cgm_trace.map((pt, i) => ({
      t: i * 5,
      iso: pt.t,
      glucose_mgdL: pt.glucose_mgdL,
      forecast_mgdL: undefined as number | undefined,
    }));
    const off = h.length;
    const f = chartPrefs.showForecast
      ? data.forecast.map((pt, i) => ({
          t: (off + i) * 5,
          iso: pt.t,
          glucose_mgdL: i === 0 ? h.at(-1)?.glucose_mgdL : undefined,
          forecast_mgdL: pt.glucose_mgdL,
        }))
      : [];
    return [...h, ...f];
  }, [data, chartPrefs.showForecast]);

  const displayName = profileNames[selectedId] || patients.find(p => p.id === selectedId)?.display_name || selectedId;
  const statusInfo  = data ? getGlucoseStatus(data.current_glucose_mgdL) : null;
  const trend       = data ? (TREND_INFO[data.trend_arrow] || { text: "Steady", icon: ArrowRight, color: "var(--emerald-600)" }) : null;

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <Activity size={22} style={{ color: "var(--emerald-600)" }} />
            Overview Dashboard
          </h2>
          <p>Monitor glucose trends and review patient simulation metrics.</p>
        </div>
        <div className="page-header-actions">
          <Badge label={data?.forecast_model || (modelMode === "hybrid" ? "Hybrid Neural-ODE Active" : "Mechanistic ODE Active")} type={modelMode === "hybrid" ? "purple" : "target"} icon={ShieldCheck} />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => fetchOverview(true)}
            disabled={refreshing || loading}
          >
            <RefreshCw size={13} className={refreshing ? "loading-spinner" : ""} />
            {refreshing ? "Refreshing..." : "Refresh Data"}
          </button>
        </div>
      </div>

      {listError && (
        <div className="error-box mb-4">
          <AlertTriangle size={16} />
          <div>{listError}</div>
        </div>
      )}

      {/* Patient Selection & Time Window Bar */}
      <div className="patient-selector-bar">
        <div style={{ flex: "1 1 220px" }}>
          <label className="form-label">Select Patient</label>
          <select
            className="form-select"
            value={selectedId}
            onChange={e => setSelectedId(e.target.value)}
            disabled={listLoading}
          >
            {listLoading ? (
              <option>Loading patients...</option>
            ) : (
              patients.map(p => (
                <option key={p.id} value={p.id}>
                  {p.display_name} ({p.category})
                </option>
              ))
            )}
          </select>
        </div>

        <div style={{ flex: "1 1 180px" }}>
          <label className="form-label">Display Alias</label>
          <input
            className="form-input"
            value={nameInput}
            onChange={e => {
              setNameInput(e.target.value);
              setNameError(e.target.value.trim().length > 35 ? "Max 35 chars" : "");
            }}
            onBlur={() => {
              const t = nameInput.trim();
              if (t && t.length <= 35) setProfileName(selectedId, t);
            }}
            maxLength={36}
            placeholder="Custom label..."
          />
          {nameError && <div className="form-error">{nameError}</div>}
        </div>

        <div>
          <label className="form-label">Forecast Engine</label>
          <div className="mode-toggle-group">
            <button
              type="button"
              className={`mode-toggle-btn ${modelMode === "mechanistic" ? "active" : ""}`}
              onClick={() => setModelMode("mechanistic")}
              title="Bergman Minimal Model 3-compartment ODE"
            >
              Mechanistic ODE
            </button>
            <button
              type="button"
              className={`mode-toggle-btn ${modelMode === "hybrid" ? "active" : ""}`}
              onClick={() => setModelMode("hybrid")}
              title="Physics-Informed Hybrid Neural-ODE (ODE + GRU Residual)"
            >
              Hybrid Neural-ODE
            </button>
          </div>
        </div>

        <div>
          <label className="form-label">Data Window</label>
          <div className="pill-group">
            {[6, 12, 24, 48].map(h => (
              <button
                key={h}
                type="button"
                className={`pill-btn ${windowHours === h ? "active" : ""}`}
                onClick={() => setWindowHours(h)}
              >
                {h}h
              </button>
            ))}
          </div>
        </div>

        {lastRefreshed && (
          <div style={{ alignSelf: "center", fontSize: 11.5, color: "var(--text-secondary)", whiteSpace: "nowrap", marginLeft: "auto" }}>
            Last sync: {lastRefreshed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
        )}
      </div>

      {dataError && (
        <div className="error-box mb-4">
          <AlertTriangle size={16} />
          <div>{dataError}</div>
        </div>
      )}

      {loading && (
        <div className="card mb-4">
          <div className="empty-state">
            <Spinner />
            <p>Loading glucose data for {displayName}...</p>
          </div>
        </div>
      )}

      {!loading && data && (
        <>
          {/* Current Glucose Card */}
          <div className={`current-glucose-card ${statusInfo?.cardCls || "target"}`}>
            <div className="glucose-main-section">
              <div className="glucose-stat-group">
                <span className="glucose-stat-label">Current Glucose</span>
                <div className="glucose-stat-val-wrap">
                  <span className="glucose-stat-number" style={{ color: statusInfo?.type === "danger" ? "var(--status-danger-text)" : statusInfo?.type === "warning" ? "var(--status-warning-text)" : "var(--emerald-800)" }}>
                    {data.current_glucose_mgdL.toFixed(1)}
                  </span>
                  <span className="glucose-stat-unit">mg/dL</span>
                </div>
              </div>

              <div className="glucose-divider" />

              <div className="glucose-stat-group">
                <span className="glucose-stat-label">Trend Direction</span>
                <div className="glucose-trend-badge">
                  {trend && <trend.icon size={20} style={{ color: trend.color }} />}
                  <span>{trend?.text}</span>
                </div>
              </div>

              <div className="glucose-divider" />

              <div className="glucose-stat-group">
                <span className="glucose-stat-label">Status</span>
                <Badge
                  label={statusInfo?.label || ""}
                  type={statusInfo?.type === "target" ? "target" : statusInfo?.type === "warning" ? "warning" : "danger"}
                />
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
              {lastRefreshed && (
                <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>
                  Last updated: {lastRefreshed.toLocaleTimeString()}
                </span>
              )}
              <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                Target Range: 70 – 180 mg/dL
              </span>
            </div>
          </div>

          {/* Key Glucose Metrics (5 Cards) */}
          <KeyMetricsRow data={data} />

          {/* Main CGM Glucose Chart */}
          <CGMChart
            chartData={chartData}
            simResponse={null}
            showForecast={chartPrefs.showForecast}
            showZones={chartPrefs.showZones}
            title={`Glucose Trend — Last ${windowHours} Hours`}
          />

          {/* Recent Events */}
          <div className="card" style={{ marginTop: 20 }}>
            <div className="card-header">
              <h3><Utensils size={15} />Recent Events &amp; Intakes</h3>
              <Badge label={`${data.meal_events.length} Events`} type="neutral" />
            </div>
            <div className="card-body" style={{ padding: data.meal_events.length > 0 ? "0" : "18px" }}>
              {data.meal_events.length === 0 ? (
                <div className="empty-state" style={{ padding: 18 }}>
                  <p>No intake or simulation events recorded in the selected {windowHours}h window.</p>
                </div>
              ) : (
                <div style={{ maxHeight: 220, overflowY: "auto" }}>
                  {data.meal_events.map((m, i) => (
                    <div
                      key={m.id || i}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "10px 18px",
                        borderBottom: "1px solid var(--border-color)",
                      }}
                    >
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-main)" }}>
                          {m.name || "Recorded Intake"}
                        </div>
                        <div style={{ fontSize: 11.5, color: "var(--text-secondary)" }}>
                          {m.category || "Dataset Meal"} · {new Date(m.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <Badge label={`${m.cho_g}g carbs`} type="neutral" />
                        {m.is_user_logged && <Badge label="User Logged" type="teal" />}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Digital Twin Architecture 4-Stage Pipeline ──────────────────────────────
function DigitalTwinArchitecturePanel() {
  return (
    <div className="card mb-4">
      <div className="card-header">
        <h3><Sliders size={15} />How the Digital Twin Works — 4-Stage Execution Pipeline</h3>
        <Badge label="Physics-Informed Architecture" type="teal" />
      </div>
      <div className="card-body">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 14 }}>
          {/* Stage 1 */}
          <div style={{ padding: 14, borderRadius: "var(--radius-sm)", background: "var(--bg-secondary)", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--emerald-800)", textTransform: "uppercase" }}>Stage 1</span>
              <span className="model-tag live">Live Data Input</span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-main)", marginBottom: 4 }}>Continuous Telemetry</div>
            <p style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Ingests 5-minute CGM glucose traces, carbohydrate meal events, and basal/bolus insulin delivery history.
            </p>
          </div>

          {/* Stage 2 */}
          <div style={{ padding: 14, borderRadius: "var(--radius-sm)", background: "var(--bg-secondary)", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--emerald-800)", textTransform: "uppercase" }}>Stage 2</span>
              <span className="model-tag live">Live ODE Sim</span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-main)", marginBottom: 4 }}>Mechanistic Minimal ODE</div>
            <p style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Bergman 3-compartment ODE numerically integrates glucose-insulin kinetics: dG/dt, dX/dt, dI/dt with patient-specific calibrated S_I and S_G.
            </p>
          </div>

          {/* Stage 3 */}
          <div style={{ padding: 14, borderRadius: "var(--radius-sm)", background: "var(--bg-secondary)", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--emerald-800)", textTransform: "uppercase" }}>Stage 3</span>
              <span className="model-tag live">Live + EKF Batch</span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-main)", marginBottom: 4 }}>Residual GRU &amp; Kalman Filter</div>
            <p style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Physics-Informed GRU learns high-frequency physiological residuals; Extended Kalman Filter reconstructs continuous hidden states with uncertainty intervals.
            </p>
          </div>

          {/* Stage 4 */}
          <div style={{ padding: 14, borderRadius: "var(--radius-sm)", background: "var(--bg-secondary)", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--emerald-800)", textTransform: "uppercase" }}>Stage 4</span>
              <span className="model-tag offline">Offline Research</span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-main)", marginBottom: 4 }}>MPC &amp; Safety Shield</div>
            <p style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Predictive control (MPC) and RL optimization evaluated with safety-shield boundaries. Strictly research; no real-world dosing connections.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── EKF State Estimation Panel ────────────────────────────────────────────────
function EKFStatePanel({ patientId, patientName }: { patientId: string; patientName: string }) {
  const [data, setData] = useState<EKFEstimateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setData(null);
    setError("");
    setLoading(false);
  }, [patientId]);

  const runEKF = () => {
    if (!patientId) return;
    setLoading(true);
    setError("");
    axios.get(`/api/patients/${patientId}/ekf_estimate`)
      .then(r => setData(r.data))
      .catch(e => setError(e.response?.data?.detail || e.message || "EKF State Estimation failed."))
      .finally(() => setLoading(false));
  };

  const traceList = data?.state_estimates || data?.trace || [];
  const nSteps = data?.n_steps ?? (Array.isArray(traceList) ? traceList.length : 0);
  const siVal = data?.estimated_sensitivity_Si ?? data?.estimated_si;
  const finalX = data?.final_state?.insulin_action_per_min ?? (Array.isArray(traceList) && traceList.length > 0 ? (traceList[traceList.length - 1]?.remote_insulin_action_est ?? traceList[traceList.length - 1]?.insulin_action_X) : null);
  const meanGlucose = data?.mean_glucose_mgdL ?? (Array.isArray(traceList) && traceList.length > 0 ? (traceList.reduce((acc: number, r: any) => acc + (r.glucose_est_mgdL || r.estimated_glucose_mgdL || 0), 0) / traceList.length).toFixed(1) : "—");

  const chartData = useMemo(() => {
    if (!Array.isArray(traceList)) return [];
    return traceList.map((r: any, i: number) => ({
      t_min: typeof r.t_min === "number" ? r.t_min : i * 5,
      measured_glucose_mgdL: r.measured_glucose_mgdL ?? r.cgm_observed_mgdL ?? undefined,
      estimated_glucose_mgdL: r.estimated_glucose_mgdL ?? r.glucose_est_mgdL ?? undefined,
      ci_lower_mgdL: r.ci_lower_mgdL ?? r.glucose_ci_lower_mgdL ?? undefined,
      ci_upper_mgdL: r.ci_upper_mgdL ?? r.glucose_ci_upper_mgdL ?? undefined,
    }));
  }, [traceList]);

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <div className="card-header">
        <h3><Sliders size={15} />Extended Kalman Filter (EKF) State Estimator</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {data && <Badge label={data.filter_status || "EKF Filter Active"} type="target" icon={CheckCircle2} />}
          <button
            type="button"
            className="btn btn-primary"
            onClick={runEKF}
            disabled={loading}
            style={{ fontSize: 12, padding: "5px 12px" }}
          >
            {loading ? <><Spinner /> Running EKF...</> : <><Zap size={13} /> {data ? "Re-run EKF Filter" : "Run EKF State Estimation"}</>}
          </button>
        </div>
      </div>
      <div className="card-body">
        <p style={{ fontSize: 12.5, color: "var(--text-secondary)", marginBottom: 12, lineHeight: 1.5 }}>
          The Extended Kalman Filter (EKF) performs retrospective batch state estimation on {patientName}'s historical window,
          estimating hidden physiological states: Plasma Glucose G(t), Remote Insulin Action X(t), and Plasma Insulin I(t) with continuous ±1.96σ confidence intervals.
        </p>

        {error && (
          <div className="error-box mb-3">
            <AlertTriangle size={15} />
            <span>{error}</span>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={runEKF}
              style={{ marginLeft: "auto", fontSize: 11, padding: "2px 8px" }}
            >
              Retry
            </button>
          </div>
        )}

        {!data && !loading && !error && (
          <div className="empty-state" style={{ padding: "16px 0" }}>
            <p>Click "Run EKF State Estimation" to compute continuous physiological state tracking with uncertainty bands.</p>
          </div>
        )}

        {data && (
          <div>
            <div className="metrics-row" style={{ marginBottom: 16 }}>
              <div className="metric-card">
                <div className="metric-icon-wrap target">
                  <Activity size={15} />
                </div>
                <div className="metric-info">
                  <div className="metric-label">Estimated Sensitivity (S_I)</div>
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {typeof siVal === "number" ? siVal.toExponential(3) : "—"}
                  </div>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon-wrap neutral">
                  <Clock size={15} />
                </div>
                <div className="metric-info">
                  <div className="metric-label">Estimation Window</div>
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {nSteps} points ({data.window_hours || 24}h)
                  </div>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon-wrap neutral">
                  <Target size={15} />
                </div>
                <div className="metric-info">
                  <div className="metric-label">Filtered Mean Glucose</div>
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {meanGlucose} mg/dL
                  </div>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon-wrap purple">
                  <Zap size={15} />
                </div>
                <div className="metric-info">
                  <div className="metric-label">Remote Action X(t)</div>
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {typeof finalX === "number" ? `${finalX.toFixed(5)} min⁻¹` : "—"}
                  </div>
                </div>
              </div>
            </div>

            {/* Continuous EKF State Chart */}
            {chartData.length > 0 && (
              <div style={{ height: 260, width: "100%", marginTop: 10 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={chartData}
                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="t_min"
                      tickFormatter={t => `${(t / 60).toFixed(0)}h`}
                      fontSize={11}
                      stroke="#64748b"
                    />
                    <YAxis domain={[40, 'auto']} fontSize={11} stroke="#64748b" />
                    <Tooltip
                      formatter={(val: any, name: any) => [
                        typeof val === "number" ? `${val.toFixed(1)} mg/dL` : val,
                        name === 'measured_glucose_mgdL' ? 'Measured CGM' :
                        name === 'estimated_glucose_mgdL' ? 'EKF Estimated G(t)' :
                        name === 'ci_upper_mgdL' ? '95% CI Upper' : '95% CI Lower'
                      ]}
                      labelFormatter={t => `Time: ${(Number(t) / 60).toFixed(1)}h (${t} min)`}
                    />
                    <ReferenceLine y={70} stroke="#dc2626" strokeDasharray="4 4" />
                    <ReferenceLine y={180} stroke="#d97706" strokeDasharray="4 4" />
                    <Area
                      type="monotone"
                      dataKey="ci_upper_mgdL"
                      stroke="transparent"
                      fill="#a7f3d0"
                      fillOpacity={0.35}
                      name="95% CI Upper"
                    />
                    <Area
                      type="monotone"
                      dataKey="ci_lower_mgdL"
                      stroke="transparent"
                      fill="#ffffff"
                      fillOpacity={1}
                      name="95% CI Lower"
                    />
                    <Area
                      type="monotone"
                      dataKey="measured_glucose_mgdL"
                      stroke="#047857"
                      strokeWidth={2}
                      fill="transparent"
                      dot={false}
                      name="Measured CGM"
                    />
                    <Area
                      type="monotone"
                      dataKey="estimated_glucose_mgdL"
                      stroke="#0284c7"
                      strokeWidth={2}
                      strokeDasharray="3 3"
                      fill="transparent"
                      dot={false}
                      name="EKF Estimated G(t)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, fontSize: 11.5, color: "var(--text-secondary)" }}>
              <span>Legend: Solid Green = CGM Observation | Blue Dashed = EKF State G(t) | Green Band = 95% Confidence Band (±1.96σ)</span>
              <span>* {DISCLAIMER}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Digital Twin Page ─────────────────────────────────────────────────────────
function DigitalTwinPage({
  selectedId,
  setSelectedId,
  chartPrefs,
  defaultWindowHours,
}: {
  selectedId: string;
  setSelectedId: (id: string) => void;
  chartPrefs: { showArea: boolean; showForecast: boolean; showZones: boolean };
  defaultWindowHours: number;
}) {
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [profile, setProfile] = useState<PatientListItem | null>(null);
  const [params, setParams] = useState<PatientParameters | null>(null);
  const [data, setData] = useState<OverviewData | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState("");
  const [simState, setSimState] = useState<"idle" | "running" | "paused" | "done">("idle");
  const [simDuration, setSimDuration] = useState(12);
  const [simResponse, setSimResponse] = useState<LiveSimulationResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState("");

  const fetchPatients = useCallback(() => {
    setPatientsLoading(true);
    axios.get("/api/patients")
      .then(r => {
        const list: PatientListItem[] = r.data.patients;
        setPatients(list);
        if (!selectedId && list.length > 0) setSelectedId(list[0].id);
      })
      .catch(() => {})
      .finally(() => setPatientsLoading(false));
  }, [selectedId, setSelectedId]);

  useEffect(() => { fetchPatients(); }, [fetchPatients]);

  const displayName = profile?.display_name ?? patients.find(p => p.id === selectedId)?.display_name ?? selectedId;

  useEffect(() => {
    if (!selectedId) return;
    const found = patients.find(p => p.id === selectedId);
    if (found) setProfile(found);
    axios.get(`/api/patients/${selectedId}/parameters`)
      .then(r => setParams(r.data))
      .catch(() => setParams(null));
  }, [selectedId, patients]);

  useEffect(() => {
    if (!selectedId) return;
    setDataLoading(true);
    setDataError("");
    axios.get(`/api/overview/${selectedId}`, { params: { window_hours: defaultWindowHours } })
      .then(r => setData(r.data))
      .catch(() => setDataError("Could not load CGM telemetry for this patient."))
      .finally(() => setDataLoading(false));
  }, [selectedId, defaultWindowHours]);

  useEffect(() => {
    setSimState("idle");
    setSimResponse(null);
    setSimError("");
  }, [selectedId]);

  const handleStartSim = async () => {
    setSimLoading(true);
    setSimError("");
    try {
      const r = await axios.post("/api/simulation/run", {
        patient_id: selectedId,
        duration_hours: simDuration,
        step_minutes: 5,
        include_user_meals: true,
      });
      setSimResponse(r.data);
      setSimState("done");
    } catch (e: any) {
      setSimError(e.response?.data?.detail || e.message || "Simulation failed.");
      setSimState("idle");
    } finally {
      setSimLoading(false);
    }
  };

  const chartData = useMemo((): ChartPt[] => {
    if (!data) return [];
    const history: ChartPt[] = data.cgm_trace.map((pt, i) => ({
      t: i * 5,
      iso: pt.t,
      glucose_mgdL: pt.glucose_mgdL,
    }));
    const offset = history.length;
    const fcast: ChartPt[] = chartPrefs.showForecast
      ? data.forecast.map((pt, i) => ({
          t: (offset + i) * 5,
          iso: pt.t,
          forecast_mgdL: pt.glucose_mgdL,
          glucose_mgdL: i === 0 ? history.at(-1)?.glucose_mgdL : undefined,
        }))
      : [];
    if (simResponse) {
      const simMap: Record<number, number> = {};
      simResponse.trace.forEach(s => { simMap[Math.round(s.t_min)] = s.glucose_mgdL; });
      history.forEach((pt, i) => {
        const tMin = i * 5;
        if (simMap[tMin] !== undefined) pt.sim_glucose_mgdL = simMap[tMin];
      });
    }
    return [...history, ...fcast];
  }, [data, simResponse, chartPrefs.showForecast]);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <HeartPulse size={22} style={{ color: "var(--emerald-600)" }} />
            My Digital Twin — {displayName}
          </h2>
          <p>Bergman Minimal Model · Mechanistic Simulation · Synthetic UVA/Padova Data</p>
        </div>
        <div className="page-header-actions">
          <Badge label="ODE Engine Active" type="target" icon={Zap} />
        </div>
      </div>

      {/* 4-Stage Architectural Pipeline */}
      <DigitalTwinArchitecturePanel />

      <div className="dashboard-grid">
        {/* Left column */}
        <div className="dashboard-left">
          {dataError && (
            <div className="error-box">
              <AlertTriangle size={16} />
              <span>{dataError}</span>
            </div>
          )}
          {simError && (
            <div className="error-box">
              <AlertTriangle size={16} />
              <span>{simError}</span>
            </div>
          )}

          {dataLoading ? (
            <div className="card">
              <div className="empty-state">
                <Spinner />
                <p>Loading patient telemetry...</p>
              </div>
            </div>
          ) : (
            <CGMChart
              chartData={chartData}
              simResponse={simResponse}
              showForecast={chartPrefs.showForecast}
              showZones={chartPrefs.showZones}
              title={`CGM Telemetry & Simulation — ${displayName}`}
            />
          )}

          <KeyMetricsRow data={data} simResponse={simResponse} />
          <HealthInsights data={data} simResponse={simResponse} />
          <MealLogger patientId={selectedId} patientName={displayName} />
          <EKFStatePanel patientId={selectedId} patientName={displayName} />

          {simResponse && (
            <div className="card">
              <div className="card-header">
                <h3><Zap size={15} />Simulation Summary</h3>
                <Badge label="Bergman Minimal Model" type="teal" />
              </div>
              <div className="card-body">
                <div className="metrics-row" style={{ marginBottom: 12 }}>
                  {[
                    { label: "Initial Glucose", val: `${simResponse.summary.initial_glucose_mgdL} mg/dL`, Icon: Activity, cls: "neutral" },
                    { label: "Peak Glucose", val: `${simResponse.summary.peak_glucose_mgdL} mg/dL`, Icon: TrendingUp, cls: simResponse.summary.peak_glucose_mgdL > 180 ? "warning" : "success" },
                    { label: "Simulated TIR", val: `${simResponse.summary.tir_pct}%`, Icon: Target, cls: simResponse.summary.tir_pct >= 70 ? "success" : "warning" },
                    { label: "Carbs Modeled", val: `${simResponse.summary.total_carbs_g}g`, Icon: Utensils, cls: "neutral" },
                  ].map(t => (
                    <div key={t.label} className="metric-card">
                      <div className={`metric-icon-wrap ${t.cls}`}>
                        <t.Icon size={16} />
                      </div>
                      <div className="metric-info">
                        <div className="metric-label">{t.label}</div>
                        <div className="metric-value">{t.val}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                  * S_I = {simResponse.parameters.Si?.toFixed?.(5) ?? "—"}, S_G = {simResponse.parameters.Sg?.toFixed?.(4) ?? "—"} · {DISCLAIMER}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="dashboard-right">
          <PatientListPanel
            patients={patients}
            selectedId={selectedId}
            onSelect={id => { setSelectedId(id); setSimState("idle"); setSimResponse(null); }}
            onRefresh={fetchPatients}
            loading={patientsLoading}
          />
          <ProfileCard profile={profile} params={params} />
          <SimulationControls
            simState={simState}
            simDuration={simDuration}
            setSimDuration={setSimDuration}
            onStart={handleStartSim}
            onPause={() => setSimState("paused")}
            onResume={() => setSimState("running")}
            onReset={() => { setSimState("idle"); setSimResponse(null); setSimError(""); }}
            loading={simLoading}
          />
          <HealthAssistant
            patientId={selectedId}
            patientName={displayName}
            metrics={data?.metrics}
            simulationSummary={simResponse?.summary}
          />
        </div>
      </div>
    </div>
  );
}

// ── What-If Lab Page with Sleep & Exercise Scenario Extensions ─────────────────
function WhatIfPage() {
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("synthetic_000");
  const [modelMode, setModelMode] = useState<"mechanistic" | "hybrid">("mechanistic");

  const [form, setForm] = useState({
    scenarioName: "Standard Meal",
    duration: 4,
    mealCho: 40,
    mealTime: 1.0,
    basalInsulin: 15,
    bolusInsulin: 0,
    sleepDuration: 8,
    sleepQuality: "Average",
    sleepBedtime: 0.0,
    exerciseType: "None",
    exerciseDuration: 0,
    exerciseIntensity: "None",
    exerciseStartTime: 1.5,
  });

  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [previousResult, setPreviousResult] = useState<WhatIfResult | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get("/api/patients")
      .then(r => {
        const pts: PatientListItem[] = r.data.patients;
        setPatients(pts);
        if (pts.length > 0) setSelectedPatientId(pts[0].id);
      })
      .catch(() => {});
  }, []);

  const PRESETS = [
    {
      name: "Standard Meal",
      mealCho: 40,
      mealTime: 1.0,
      basalInsulin: 15,
      bolusInsulin: 0,
      duration: 4,
      sleepDuration: 8,
      sleepQuality: "Average",
      exerciseType: "None",
      exerciseDuration: 0,
      exerciseIntensity: "None",
    },
    {
      name: "High-Carbohydrate Meal",
      mealCho: 75,
      mealTime: 1.0,
      basalInsulin: 15,
      bolusInsulin: 0,
      duration: 6,
      sleepDuration: 8,
      sleepQuality: "Good",
      exerciseType: "None",
      exerciseDuration: 0,
      exerciseIntensity: "None",
    },
    {
      name: "Light Activity Scenario",
      mealCho: 30,
      mealTime: 0.5,
      basalInsulin: 15,
      bolusInsulin: 0,
      duration: 4,
      sleepDuration: 7.5,
      sleepQuality: "Good",
      exerciseType: "Walking",
      exerciseDuration: 30,
      exerciseIntensity: "Light",
    },
    {
      name: "Moderate Activity Scenario",
      mealCho: 45,
      mealTime: 1.0,
      basalInsulin: 15,
      bolusInsulin: 100,
      duration: 5,
      sleepDuration: 8,
      sleepQuality: "Good",
      exerciseType: "Running",
      exerciseDuration: 45,
      exerciseIntensity: "Moderate",
    },
    {
      name: "Reduced Sleep Scenario",
      mealCho: 50,
      mealTime: 1.0,
      basalInsulin: 15,
      bolusInsulin: 0,
      duration: 6,
      sleepDuration: 5,
      sleepQuality: "Poor",
      exerciseType: "None",
      exerciseDuration: 0,
      exerciseIntensity: "None",
    },
    {
      name: "Combined Meal & Activity",
      mealCho: 60,
      mealTime: 1.0,
      basalInsulin: 15,
      bolusInsulin: 150,
      duration: 6,
      sleepDuration: 7,
      sleepQuality: "Average",
      exerciseType: "Cycling",
      exerciseDuration: 45,
      exerciseIntensity: "Moderate",
    },
  ];

  const applyPreset = (p: typeof PRESETS[0]) => {
    setForm(prev => ({
      ...prev,
      scenarioName: p.name,
      duration: p.duration,
      mealCho: p.mealCho,
      mealTime: p.mealTime,
      basalInsulin: p.basalInsulin,
      bolusInsulin: p.bolusInsulin,
      sleepDuration: p.sleepDuration,
      sleepQuality: p.sleepQuality,
      exerciseType: p.exerciseType,
      exerciseDuration: p.exerciseDuration,
      exerciseIntensity: p.exerciseIntensity,
    }));
  };

  const run = async () => {
    setLoading(true);
    setError("");
    try {
      if (result) {
        setPreviousResult(result);
      }
      const r = await axios.post("/api/whatif", {
        scenario_name: form.scenarioName,
        patient_id: selectedPatientId,
        duration_hours: form.duration,
        meal_cho_g: form.mealCho,
        meal_time_h: form.mealTime,
        basal_insulin_mU_per_min: form.basalInsulin,
        bolus_insulin_mU: form.bolusInsulin,
        sleep_duration_hours: form.sleepDuration,
        sleep_quality: form.sleepQuality,
        sleep_bedtime_h: form.sleepBedtime,
        exercise_type: form.exerciseType,
        exercise_duration_min: form.exerciseDuration,
        exercise_intensity: form.exerciseIntensity,
        exercise_start_time_h: form.exerciseStartTime,
        model_mode: modelMode,
      });
      setResult(r.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Simulation execution failed.");
    } finally {
      setLoading(false);
    }
  };

  const set = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }));

  const comparisonChartData = useMemo(() => {
    if (!result) return [];
    if (!previousResult || !showComparison) {
      return result.trace.map(pt => ({
        t_min: pt.t_min,
        sim_glucose_mgdL: pt.glucose_mgdL,
      }));
    }
    const mapPrev: Record<number, number> = {};
    previousResult.trace.forEach(pt => { mapPrev[Math.round(pt.t_min)] = pt.glucose_mgdL; });
    return result.trace.map(pt => ({
      t_min: pt.t_min,
      sim_glucose_mgdL: pt.glucose_mgdL,
      baseline_glucose_mgdL: mapPrev[Math.round(pt.t_min)],
    }));
  }, [result, previousResult, showComparison]);

  const selectedPatientName = patients.find(p => p.id === selectedPatientId)?.display_name || selectedPatientId;

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <FlaskConical size={22} style={{ color: "var(--emerald-600)" }} />
            What-If Scenario Lab
          </h2>
          <p>Simulate meal absorption, sleep context, and exercise dynamics using the Bergman Minimal Model.</p>
        </div>
        <div className="page-header-actions">
          <Badge label="In Silico Experimentation" type="teal" icon={Sliders} />
        </div>
      </div>

      <div className="info-callout mb-4">
        <FlaskConical size={18} className="info-callout-icon" />
        <div className="info-callout-text">
          <h4>Research Simulation Environment</h4>
          <p>
            The Bergman Minimal Model numerically simulates glucose-insulin kinetics from carbohydrate ingestion and basal insulin delivery.
            Sleep and exercise inputs are captured as research scenario metadata. All outputs are synthetic trajectories for research analysis.
          </p>
        </div>
      </div>

      <div className="grid-2">
        {/* Left Column: Simulation Inputs */}
        <div className="card">
          <div className="card-header">
            <h3><Sliders size={15} />Simulation Inputs</h3>
            <Badge label="Bergman ODE Model" type="neutral" />
          </div>
          <div className="card-body">
            {/* Patient Selector & Engine Mode */}
            <div className="form-group">
              <label className="form-label">Virtual Patient Profile</label>
              <select
                className="form-select"
                value={selectedPatientId}
                onChange={e => setSelectedPatientId(e.target.value)}
              >
                {patients.map(p => (
                  <option key={p.id} value={p.id}>{p.display_name} ({p.category})</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Simulation Engine Mode</label>
              <div className="mode-toggle-group" style={{ width: "100%" }}>
                <button
                  type="button"
                  className={`mode-toggle-btn ${modelMode === "mechanistic" ? "active" : ""}`}
                  onClick={() => setModelMode("mechanistic")}
                  style={{ flex: 1 }}
                >
                  Mechanistic Minimal ODE
                </button>
                <button
                  type="button"
                  className={`mode-toggle-btn ${modelMode === "hybrid" ? "active" : ""}`}
                  onClick={() => setModelMode("hybrid")}
                  style={{ flex: 1 }}
                >
                  Hybrid Neural-ODE (PiNN)
                </button>
              </div>
            </div>

            {/* Quick Presets */}
            <div className="form-group">
              <label className="form-label">Scenario Presets</label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {PRESETS.map(p => (
                  <button
                    key={p.name}
                    type="button"
                    className="assistant-quick-btn"
                    onClick={() => applyPreset(p)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Section A: Core Scenario & Meal */}
            <div className="section-divider-title">
              <span>A. Scenario &amp; Meal Inputs</span>
              <Badge label="Modeled via ODE" type="target" />
            </div>

            <div className="form-group">
              <label className="form-label">Scenario Name</label>
              <input
                className="form-input"
                value={form.scenarioName}
                onChange={e => set("scenarioName", e.target.value)}
              />
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Duration (hours)</label>
                <input
                  className="form-input"
                  type="number"
                  min={1}
                  max={24}
                  step={0.5}
                  value={form.duration}
                  onChange={e => set("duration", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Meal Time (h into sim)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={23}
                  step={0.25}
                  value={form.mealTime}
                  onChange={e => set("mealTime", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Carbohydrates (g)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={200}
                  step={5}
                  value={form.mealCho}
                  onChange={e => set("mealCho", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Basal Insulin (mU/min)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={100}
                  step={1}
                  value={form.basalInsulin}
                  onChange={e => set("basalInsulin", Number(e.target.value))}
                />
              </div>
            </div>

            {/* Section B: Sleep Information */}
            <div className="section-divider-title">
              <span>B. Sleep Scenario</span>
              <Badge label="Experimental Metadata" type="neutral" />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Sleep Duration (hours)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={24}
                  step={0.5}
                  value={form.sleepDuration}
                  onChange={e => set("sleepDuration", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Sleep Quality</label>
                <select
                  className="form-select"
                  value={form.sleepQuality}
                  onChange={e => set("sleepQuality", e.target.value)}
                >
                  <option>Poor</option>
                  <option>Average</option>
                  <option>Good</option>
                </select>
              </div>
            </div>

            {/* Section C: Exercise Information */}
            <div className="section-divider-title">
              <span>C. Exercise Scenario</span>
              <Badge label="Experimental Metadata" type="neutral" />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Activity Type</label>
                <select
                  className="form-select"
                  value={form.exerciseType}
                  onChange={e => set("exerciseType", e.target.value)}
                >
                  <option>None</option>
                  <option>Walking</option>
                  <option>Running</option>
                  <option>Cycling</option>
                  <option>Resistance Training</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Duration (minutes)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={180}
                  step={5}
                  value={form.exerciseDuration}
                  onChange={e => set("exerciseDuration", Number(e.target.value))}
                  disabled={form.exerciseType === "None"}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Intensity</label>
                <select
                  className="form-select"
                  value={form.exerciseIntensity}
                  onChange={e => set("exerciseIntensity", e.target.value)}
                  disabled={form.exerciseType === "None"}
                >
                  <option>None</option>
                  <option>Light</option>
                  <option>Moderate</option>
                  <option>Vigorous</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Start Time (h)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  max={23}
                  step={0.5}
                  value={form.exerciseStartTime}
                  onChange={e => set("exerciseStartTime", Number(e.target.value))}
                  disabled={form.exerciseType === "None"}
                />
              </div>
            </div>

            <button
              className="btn btn-primary w-full"
              style={{ marginTop: 14 }}
              onClick={run}
              disabled={loading}
            >
              {loading ? <><Spinner /> Simulating ODE Dynamics...</> : <><Play size={14} /> Run What-If Simulation</>}
            </button>

            {error && (
              <div className="error-box" style={{ marginTop: 12 }}>
                <AlertTriangle size={15} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Simulation Results */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {result ? (
            <>
              {/* Simulation Status Header Card */}
              <div className="card">
                <div className="card-header">
                  <h3>
                    <Activity size={15} />
                    {result.scenario_name} — Glucose Response
                  </h3>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    {previousResult && (
                      <button
                        type="button"
                        className={`btn ${showComparison ? "btn-primary" : "btn-secondary"}`}
                        style={{ padding: "4px 9px", fontSize: 12 }}
                        onClick={() => setShowComparison(!showComparison)}
                      >
                        <ArrowLeftRight size={13} />
                        {showComparison ? "Hide Comparison" : "Compare with Previous"}
                      </button>
                    )}
                    <Badge label="ODE Simulation" type="teal" icon={ShieldCheck} />
                  </div>
                </div>
                <div className="card-body">
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginBottom: 12 }}>
                    Patient: <strong>{selectedPatientName}</strong> · Duration: <strong>{result.duration_hours}h</strong> · Model: <strong>Bergman Minimal ODE</strong>
                  </div>

                  {/* Glucose Trajectory Chart */}
                  <div className="chart-container" style={{ height: 260 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={comparisonChartData} margin={{ top: 6, right: 12, left: -14, bottom: 0 }}>
                        <defs>
                          <linearGradient id="whatifSimGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={CHART_COLORS.odeSimulation} stopOpacity={0.22} />
                            <stop offset="95%" stopColor={CHART_COLORS.odeSimulation} stopOpacity={0.0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 4" stroke="#E2E8F0" vertical={false} />
                        <XAxis
                          dataKey="t_min"
                          stroke="#94A3B8"
                          tick={{ fontSize: 11, fill: "#64748B" }}
                          tickFormatter={tLabel}
                        />
                        <YAxis
                          domain={[40, 320]}
                          stroke="#94A3B8"
                          tick={{ fontSize: 11, fill: "#64748B" }}
                          tickCount={7}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#FFFFFF",
                            border: "1px solid #E2E8F0",
                            borderRadius: 8,
                            fontSize: 12.5,
                          }}
                          formatter={(v: any, name: any) => [
                            `${Number(v).toFixed(1)} mg/dL`,
                            name === "sim_glucose_mgdL" ? "Current Scenario (ODE)" : "Previous Run (Baseline)",
                          ]}
                          labelFormatter={(t: any) => `Time = ${tLabel(Number(t))}`}
                        />
                        <ReferenceArea y1={70} y2={180} fill={CHART_COLORS.targetZone} fillOpacity={0.75} stroke="none" />
                        <ReferenceLine
                          y={70}
                          stroke={CHART_COLORS.lowThresh}
                          strokeDasharray="4 3"
                          strokeWidth={1.5}
                          label={{ value: "70", fill: CHART_COLORS.lowThresh, fontSize: 10, position: "insideBottomLeft" }}
                        />
                        <ReferenceLine
                          y={180}
                          stroke={CHART_COLORS.highThresh}
                          strokeDasharray="4 3"
                          strokeWidth={1.5}
                          label={{ value: "180", fill: CHART_COLORS.highThresh, fontSize: 10, position: "insideTopLeft" }}
                        />
                        {/* Current Scenario Trace (Teal Solid) */}
                        <Area
                          type="monotone"
                          dataKey="sim_glucose_mgdL"
                          stroke={CHART_COLORS.odeSimulation}
                          strokeWidth={2.5}
                          fill="url(#whatifSimGrad)"
                          dot={false}
                          activeDot={{ r: 4 }}
                          name="sim_glucose_mgdL"
                        />
                        {/* Optional Comparison Trace (Grey Dashed) */}
                        {showComparison && (
                          <Area
                            type="monotone"
                            dataKey="baseline_glucose_mgdL"
                            stroke="#64748B"
                            strokeWidth={2}
                            strokeDasharray="4 4"
                            fill="none"
                            dot={false}
                            name="baseline_glucose_mgdL"
                          />
                        )}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <ChartLegend
                    hasSimulation={true}
                    hasBaseline={showComparison}
                    showZones={true}
                  />

                  {/* Event Timeline Markers */}
                  <div className="event-timeline-grid">
                    <div className="event-timeline-card">
                      <div className="event-timeline-header">
                        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                          <Utensils size={13} style={{ color: "var(--emerald-600)" }} /> Meal Intake
                        </span>
                        <Badge label="ODE Input" type="target" />
                      </div>
                      <div className="event-timeline-val">{result.meal_cho_g}g Carbohydrates</div>
                      <div className="event-timeline-note">Ingested at t = {result.meal_time_h}h</div>
                    </div>

                    <div className="event-timeline-card">
                      <div className="event-timeline-header">
                        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                          <Moon size={13} style={{ color: "var(--purple-600)" }} /> Sleep Scenario
                        </span>
                        <Badge label="Metadata" type="neutral" />
                      </div>
                      <div className="event-timeline-val">{result.sleep_scenario?.duration_hours ?? 8}h · {result.sleep_scenario?.quality ?? "Average"}</div>
                      <div className="event-timeline-note">Recorded context</div>
                    </div>

                    <div className="event-timeline-card">
                      <div className="event-timeline-header">
                        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                          <Activity size={13} style={{ color: "var(--teal-600)" }} /> Exercise
                        </span>
                        <Badge label="Metadata" type="neutral" />
                      </div>
                      <div className="event-timeline-val">
                        {result.exercise_scenario?.type !== "None"
                          ? `${result.exercise_scenario?.type} (${result.exercise_scenario?.duration_min} min)`
                          : "No Exercise"}
                      </div>
                      <div className="event-timeline-note">
                        {result.exercise_scenario?.type !== "None" ? `Intensity: ${result.exercise_scenario?.intensity}` : "Resting scenario"}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Computed Metrics Cards */}
              <div className="card">
                <div className="card-header">
                  <h3><BarChart3 size={15} />Key Computed Simulation Metrics</h3>
                </div>
                <div className="card-body">
                  <div className="metrics-row" style={{ marginBottom: 0 }}>
                    {[
                      {
                        label: "Initial Glucose",
                        val: `${result.metrics.initial_glucose_mgdL ?? "—"} mg/dL`,
                        Icon: Activity,
                        cls: "neutral",
                      },
                      {
                        label: "Peak Glucose",
                        val: `${result.metrics.peak_glucose_mgdL} mg/dL`,
                        Icon: TrendingUp,
                        cls: result.metrics.peak_glucose_mgdL > 180 ? "warning" : "success",
                      },
                      {
                        label: "Time to Peak",
                        val: result.metrics.time_to_peak_h != null ? `${(result.metrics.time_to_peak_h * 60).toFixed(0)} min` : "—",
                        Icon: Clock,
                        cls: "neutral",
                      },
                      {
                        label: "Final Glucose",
                        val: `${result.metrics.final_glucose_mgdL ?? "—"} mg/dL`,
                        Icon: Activity,
                        cls: "neutral",
                      },
                      {
                        label: "Net Change",
                        val: result.metrics.glucose_change_mgdL != null ? `${result.metrics.glucose_change_mgdL > 0 ? "+" : ""}${result.metrics.glucose_change_mgdL} mg/dL` : "—",
                        Icon: TrendingUp,
                        cls: "neutral",
                      },
                      {
                        label: "Simulated TIR",
                        val: `${result.metrics.tir_pct}%`,
                        Icon: Target,
                        cls: result.metrics.tir_pct >= 70 ? "success" : "warning",
                      },
                    ].map(t => (
                      <div key={t.label} className="metric-card">
                        <div className={`metric-icon-wrap ${t.cls}`}>
                          <t.Icon size={16} />
                        </div>
                        <div className="metric-info">
                          <div className="metric-label">{t.label}</div>
                          <div className="metric-value" style={{ fontSize: 18 }}>{t.val}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* What This Simulation Shows Interpretation */}
              <div className="card">
                <div className="card-header">
                  <h3><Info size={15} />What This Simulation Shows</h3>
                </div>
                <div className="card-body">
                  <p style={{ fontSize: 13.5, color: "var(--text-main)", lineHeight: 1.6 }}>
                    {result.interpretation}
                  </p>
                  <div style={{ marginTop: 10, fontSize: 11.5, color: "var(--text-secondary)" }}>
                    * In silico mechanistic simulation via Bergman Minimal Model ODE. Sleep and exercise factors are logged as experimental metadata.
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="card" style={{ minHeight: 300 }}>
              <div className="empty-state">
                <FlaskConical size={38} style={{ color: "var(--emerald-600)" }} />
                <h4 style={{ fontWeight: 600, color: "var(--text-main)", marginTop: 6 }}>Ready for Scenario Simulation</h4>
                <p style={{ maxWidth: 420 }}>
                  Configure virtual patient parameters, meal carbohydrate load, sleep, and activity scenarios on the left and click "Run What-If Simulation".
                </p>
              </div>
            </div>
          )}

          {/* Integrated Assistant */}
          <HealthAssistant
            patientId={selectedPatientId}
            patientName={selectedPatientName}
            metrics={result?.metrics}
            scenarioInfo={{ meal_cho_g: form.mealCho, duration_hours: form.duration }}
            sleepInfo={{ duration_hours: form.sleepDuration, quality: form.sleepQuality }}
            exerciseInfo={{ type: form.exerciseType, duration_min: form.exerciseDuration, intensity: form.exerciseIntensity }}
          />
        </div>
      </div>
    </div>
  );
}

// ── Model Comparison & Benchmark Page ──────────────────────────────────────
function ModelComparisonPage() {
  const [data, setData] = useState<ModelComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedHorizon, setSelectedHorizon] = useState<number | "all">("all");

  const fetchBenchmark = useCallback(() => {
    setLoading(true);
    setError("");
    axios.get("/api/models/comparison")
      .then(r => setData(r.data))
      .catch(e => setError(e.response?.data?.detail || e.message || "Failed to load model comparison metrics."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchBenchmark();
  }, [fetchBenchmark]);

  const rawHorizons = data?.horizons || [30, 60, 120, 240];
  const horizons: number[] = Array.isArray(rawHorizons)
    ? rawHorizons.map((h: any) => typeof h === "number" ? h : parseInt(String(h), 10)).filter((n: number) => !isNaN(n))
    : [30, 60, 120, 240];

  const models = Array.isArray(data?.models) ? data.models : [];
  const records = Array.isArray(data?.records) ? data.records : [];
  const findings = Array.isArray(data?.summary_findings) ? data.summary_findings : (Array.isArray(data?.key_takeaways) ? data.key_takeaways : []);

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <BarChart3 size={22} style={{ color: "var(--emerald-600)" }} />
            Model Benchmark &amp; Evaluation Center
          </h2>
          <p>Strictly controlled, held-out evaluation on synthetic_004 across 30, 60, 120, and 240-minute prediction horizons.</p>
        </div>
        <div className="page-header-actions">
          <Badge label="Held-Out Test Set" type="target" icon={ShieldCheck} />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={fetchBenchmark}
            disabled={loading}
          >
            <RefreshCw size={13} className={loading ? "loading-spinner" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="error-box mb-4">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={fetchBenchmark}
            style={{ marginLeft: "auto", fontSize: 11, padding: "2px 8px" }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Dataset & Protocol Information */}
      <div className="info-callout mb-4">
        <Info size={18} className="info-callout-icon" />
        <div className="info-callout-text">
          <h4>Held-Out Evaluation Methodology</h4>
          <p>
            {data?.evaluation_dataset || "Held-out Test Patient: synthetic_004 (12-day continuous UVA/Padova trace)"}.
            {data?.protocol ? ` ${data.protocol}` : " Evaluated using rolling-window inference with zero temporal or patient leakage."} All metrics correspond to verified Phase 4 experiment reports.
          </p>
        </div>
      </div>

      {/* Horizon Filter Bar */}
      <div className="patient-selector-bar mb-4">
        <div>
          <label className="form-label">Prediction Horizon Filter</label>
          <div className="pill-group">
            <button
              type="button"
              className={`pill-btn ${selectedHorizon === "all" ? "active" : ""}`}
              onClick={() => setSelectedHorizon("all")}
            >
              All Horizons
            </button>
            {horizons.map(h => (
              <button
                key={h}
                type="button"
                className={`pill-btn ${selectedHorizon === h ? "active" : ""}`}
                onClick={() => setSelectedHorizon(h)}
              >
                +{h} min
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="card mb-4">
          <div className="empty-state">
            <Spinner />
            <p>Loading held-out benchmark results...</p>
          </div>
        </div>
      ) : (models.length > 0 || records.length > 0) ? (
        <>
          {/* Main Benchmark Comparison Table */}
          <div className="card mb-4">
            <div className="card-header">
              <h3><Target size={15} />Comparative Prediction Performance</h3>
              <Badge label={`${models.length > 0 ? models.length : 4} Evaluated Architectures`} type="neutral" />
            </div>
            <div className="card-body" style={{ padding: 0 }}>
              <div className="benchmark-table-container">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Model Architecture</th>
                      <th>Category</th>
                      <th>Execution Status</th>
                      <th>Horizon</th>
                      <th>RMSE (mg/dL)</th>
                      <th>MAE (mg/dL)</th>
                      <th>MARD (%)</th>
                      <th>Clarke Zone A+B (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.length > 0 ? (
                      models.flatMap((model: any) => {
                        const modelHorizons = selectedHorizon === "all" ? horizons : [selectedHorizon as number];
                        return modelHorizons.map(h => {
                          const m = model?.horizons?.[String(h)] || model?.horizons?.[`${h}min`];
                          if (!m) return null;
                          const isBestRMSE = (h === 30 && model.model_key === "hybrid_neural_ode") ||
                                             (h === 60 && model.model_key === "hybrid_neural_ode") ||
                                             (h === 120 && model.model_key === "mechanistic_ode") ||
                                             (h === 240 && model.model_key === "mechanistic_ode");

                          return (
                            <tr key={`${model.model_key || model.model_name}-${h}`}>
                              <td style={{ fontWeight: 600, color: "var(--text-main)" }}>
                                {model.model_name || model.name || "Model"}
                              </td>
                              <td>
                                <Badge label={model.category || "Research"} type="neutral" />
                              </td>
                              <td>
                                <span className={`model-tag ${(model.execution_status || "").includes("Live") ? "live" : "offline"}`}>
                                  {model.execution_status || "Offline Benchmark"}
                                </span>
                              </td>
                              <td style={{ fontWeight: 600, color: "var(--emerald-800)" }}>
                                +{h} min
                              </td>
                              <td>
                                <span className={isBestRMSE ? "benchmark-best" : ""}>
                                  {typeof m.rmse_mgdL === "number" ? m.rmse_mgdL.toFixed(2) : (m.rmse ?? "—")}
                                </span>
                              </td>
                              <td>{typeof m.mae_mgdL === "number" ? m.mae_mgdL.toFixed(2) : (m.mae ?? "—")}</td>
                              <td>{typeof m.mard_pct === "number" ? `${m.mard_pct.toFixed(1)}%` : "—"}</td>
                              <td style={{ fontWeight: 600, color: (m.clarke_zone_a_plus_b_pct ?? m.clarke_ab ?? 0) >= 85 ? "var(--emerald-700)" : "var(--text-body)" }}>
                                {typeof m.clarke_zone_a_plus_b_pct === "number"
                                  ? `${m.clarke_zone_a_plus_b_pct.toFixed(1)}%`
                                  : (m.clarke_ab ? `${m.clarke_ab}%` : "—")}
                              </td>
                            </tr>
                          );
                        });
                      })
                    ) : (
                      records
                        .filter((r: any) => selectedHorizon === "all" || String(r.Horizon).includes(String(selectedHorizon)))
                        .map((r: any, idx: number) => (
                          <tr key={idx}>
                            <td style={{ fontWeight: 600, color: "var(--text-main)" }}>{r.Model || r.model_name}</td>
                            <td><Badge label="Research" type="neutral" /></td>
                            <td><span className="model-tag offline">Benchmark</span></td>
                            <td style={{ fontWeight: 600, color: "var(--emerald-800)" }}>{r.Horizon}</td>
                            <td><span className={r.Model?.includes("Hybrid") ? "benchmark-best" : ""}>{r.RMSE_mgdL ?? r.rmse}</span></td>
                            <td>{r.MAE_mgdL ?? r.mae}</td>
                            <td>—</td>
                            <td>{r.Clarke_AB_pct ? `${r.Clarke_AB_pct}%` : "—"}</td>
                          </tr>
                        ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key Findings & Insights Grid */}
          <div className="grid-2 mb-4">
            <div className="card">
              <div className="card-header">
                <h3><CheckCircle2 size={15} style={{ color: "var(--emerald-600)" }} />Near-Term Horizon Superiority (30 &amp; 60 min)</h3>
              </div>
              <div className="card-body">
                <p style={{ fontSize: 13, color: "var(--text-main)", lineHeight: 1.55 }}>
                  The <strong>Physics-Informed Hybrid Neural-ODE</strong> achieves the lowest prediction error across near-term windows (<strong>25.85 mg/dL RMSE</strong> at +30m and <strong>25.02 mg/dL RMSE</strong> at +60m), delivering <strong>90.1% Clarke Error Grid Zone A+B clinical accuracy</strong>.
                </p>
                <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-secondary)" }}>
                  The GRU residual model effectively captures high-frequency meal absorption lag and individual subcutaneous insulin pharmacokinetics that standard first-principles ODEs miss.
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3><ShieldCheck size={15} style={{ color: "var(--teal-600)" }} />Long-Horizon Mechanistic Anchoring (120 &amp; 240 min)</h3>
              </div>
              <div className="card-body">
                <p style={{ fontSize: 13, color: "var(--text-main)", lineHeight: 1.55 }}>
                  At longer prediction horizons (+120m and +240m), the <strong>Mechanistic Bergman ODE</strong> acts as an essential physiological anchor (<strong>28.91 mg/dL RMSE</strong>), preventing unconstrained machine-learning drift.
                </p>
                <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-secondary)" }}>
                  Pure deep learning architectures (Pure ML) degrade rapidly over extended horizons without mechanistic boundary constraints, confirming the necessity of physics-informed hybrid modeling.
                </div>
              </div>
            </div>
          </div>

          {/* Summary Findings Bulletins */}
          {findings.length > 0 && (
            <div className="card mb-4">
              <div className="card-header">
                <h3><Info size={15} />Key Scientific Conclusions from Phase 4</h3>
              </div>
              <div className="card-body">
                <ul style={{ paddingLeft: 20, fontSize: 13, color: "var(--text-main)", lineHeight: 1.65 }}>
                  {findings.map((item: any, idx: number) => (
                    <li key={idx} style={{ marginBottom: 6 }}>{String(item)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Research Disclaimer */}
          <div className="info-callout">
            <ShieldCheck size={18} className="info-callout-icon" />
            <div className="info-callout-text">
              <h4>Research Evaluation Standard</h4>
              <p>
                {data?.disclaimer || DISCLAIMER}
              </p>
            </div>
          </div>
        </>
      ) : !loading && !error ? (
        <div className="card mb-4">
          <div className="empty-state">
            <BarChart3 size={32} style={{ color: "var(--text-muted)", marginBottom: 8 }} />
            <h4>No Benchmark Metrics Available</h4>
            <p>Could not retrieve evaluation records. Click Refresh to query the backend endpoint.</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

// ── History Page ──────────────────────────────────────────────────────────────
function HistoryPage({ selectedId, patients }: { selectedId: string; patients: PatientListItem[]; }) {
  const [data, setData] = useState<PatientTrace | null>(null);
  const [loading, setLoading] = useState(false);
  const [localId, setLocalId] = useState(selectedId);

  useEffect(() => setLocalId(selectedId), [selectedId]);

  useEffect(() => {
    if (!localId) return;
    setLoading(true);
    axios.get(`/api/patients/${localId}/trace`)
      .then(r => setData(r.data))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [localId]);

  const traceData = data?.trace.map((pt, i) => ({ t: i * 5, glucose_mgdL: pt.glucose_mgdL })) ?? [];

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <HistoryIcon size={22} style={{ color: "var(--emerald-600)" }} />
            Longitudinal Telemetry History
          </h2>
          <p>Review comprehensive historical traces and summary variability metrics.</p>
        </div>
        <div className="page-header-actions">
          <Badge label="Complete Parquet Records" type="neutral" icon={ShieldCheck} />
        </div>
      </div>

      <div className="patient-selector-bar">
        <div style={{ flex: "1 1 240px" }}>
          <label className="form-label">Select Patient</label>
          <select className="form-select" value={localId} onChange={e => setLocalId(e.target.value)}>
            {patients.map(p => (
              <option key={p.id} value={p.id}>{p.display_name} ({p.category})</option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="card">
          <div className="empty-state">
            <Spinner />
            <p>Loading full trace dataset...</p>
          </div>
        </div>
      )}

      {!loading && data && (
        <>
          <div className="metrics-row mb-4">
            {[
              { label: "Overall TIR", val: `${data.metrics.tir_pct}%`, sub: "Target: ≥70%", Icon: Target, cls: data.metrics.tir_pct >= 70 ? "success" : "warning" },
              { label: "Below Range", val: `${data.metrics.tbr_pct}%`, sub: "Target: <4%", Icon: AlertTriangle, cls: data.metrics.tbr_pct > 4 ? "danger" : "success" },
              { label: "Above Range", val: `${data.metrics.tar_pct}%`, sub: "Target: <25%", Icon: TrendingUp, cls: data.metrics.tar_pct > 25 ? "warning" : "success" },
              { label: "Mean Glucose", val: `${data.metrics.mean_glucose_mgdL}`, sub: "mg/dL", Icon: BarChart3, cls: "neutral" },
              { label: "Variability (SD)", val: `${data.metrics.std_glucose_mgdL}`, sub: "mg/dL", Icon: Activity, cls: "neutral" },
            ].map(t => (
              <div key={t.label} className="metric-card">
                <div className={`metric-icon-wrap ${t.cls}`}>
                  <t.Icon size={18} />
                </div>
                <div className="metric-info">
                  <div className="metric-label">{t.label}</div>
                  <div className="metric-value">{t.val}</div>
                  <div className="metric-unit">{t.sub}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="card">
            <div className="card-header">
              <h3><HistoryIcon size={15} />Full Historical Sensor Trace — {data.n_readings} Observations</h3>
              <Badge label="Synthetic Benchmark" type="neutral" icon={ShieldCheck} />
            </div>
            <div className="card-body">
              <div className="chart-container" style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={traceData} margin={{ top: 8, right: 14, left: -14, bottom: 0 }}>
                    <defs>
                      <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.cgmHistorical} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={CHART_COLORS.cgmHistorical} stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 4" stroke="#E2E8F0" vertical={false} />
                    <XAxis
                      dataKey="t"
                      stroke="#94A3B8"
                      tick={{ fontSize: 11, fill: "#64748B" }}
                      tickFormatter={tLabel}
                      interval={Math.max(1, Math.floor(traceData.length / 8))}
                    />
                    <YAxis
                      domain={[40, 320]}
                      stroke="#94A3B8"
                      tick={{ fontSize: 11, fill: "#64748B" }}
                      tickCount={8}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#FFFFFF",
                        border: "1px solid #E2E8F0",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v: any) => [`${Number(v).toFixed(1)} mg/dL`, "Glucose"]}
                      labelFormatter={(t: any) => `Time = ${tLabel(Number(t))}`}
                    />
                    <ReferenceArea y1={70} y2={180} fill={CHART_COLORS.targetZone} fillOpacity={0.75} stroke="none" />
                    <ReferenceLine y={70}  stroke={CHART_COLORS.lowThresh} strokeDasharray="4 3" strokeWidth={1.5} />
                    <ReferenceLine y={180} stroke={CHART_COLORS.highThresh} strokeDasharray="4 3" strokeWidth={1.5} />
                    <Area
                      type="monotone"
                      dataKey="glucose_mgdL"
                      stroke={CHART_COLORS.cgmHistorical}
                      strokeWidth={2}
                      fill="url(#histGrad)"
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <ChartLegend hasHistorical={true} showZones={true} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Settings Page ─────────────────────────────────────────────────────────────
function SettingsPage({
  defaultWindowHours,
  setDefaultWindowHours,
  refreshInterval,
  setRefreshInterval,
  chartPrefs,
  setChartPrefs,
  largeText,
  setLargeText,
  onResetPreferences,
  health,
}: any) {
  const [activeTab, setActiveTab] = useState("general");
  const [phases, setPhases] = useState<ExperimentPhase[]>([]);
  const [selectedPhase, setSelectedPhase] = useState<ExperimentPhase | null>(null);
  const [report, setReport] = useState("");
  const [loadingReport, setLoadingReport] = useState(false);

  useEffect(() => {
    axios.get("/api/experiments")
      .then(r => setPhases(r.data.phases))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedPhase?.report_available) { setReport(""); return; }
    setLoadingReport(true);
    axios.get(`/api/experiments/${selectedPhase.report}/report`)
      .then(r => setReport(r.data.content))
      .catch(() => setReport("Report artifact not available."))
      .finally(() => setLoadingReport(false));
  }, [selectedPhase]);

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">
          <h2>
            <SettingsIcon size={22} style={{ color: "var(--emerald-600)" }} />
            Settings &amp; Research Configuration
          </h2>
          <p>Configure platform preferences, inspect research phases, and verify local data boundaries.</p>
        </div>
      </div>

      <div className="tabs">
        {[
          { id: "general", label: "General" },
          { id: "chart", label: "Chart Preferences" },
          { id: "research", label: "Research Artifacts" },
          { id: "privacy", label: "Privacy & Data Boundaries" },
        ].map(t => (
          <button
            key={t.id}
            type="button"
            className={`tab ${activeTab === t.id ? "active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === "general" && (
        <div className="card">
          <div className="card-header">
            <h3><SettingsIcon size={15} />General Platform Preferences</h3>
          </div>
          <div className="card-body">
            {[
              {
                label: "Default Time Window",
                desc: "Initial time duration loaded on the Overview dashboard.",
                el: (
                  <div className="pill-group">
                    {[6, 12, 24, 48].map(h => (
                      <button
                        key={h}
                        type="button"
                        className={`pill-btn ${defaultWindowHours === h ? "active" : ""}`}
                        onClick={() => setDefaultWindowHours(h)}
                      >
                        {h}h
                      </button>
                    ))}
                  </div>
                ),
              },
              {
                label: "Auto-Refresh Interval",
                desc: "Polling cadence for updating overview telemetry.",
                el: (
                  <select
                    className="form-select"
                    style={{ width: 140 }}
                    value={refreshInterval}
                    onChange={e => setRefreshInterval(Number(e.target.value))}
                  >
                    {[0, 30, 60, 120, 300].map(v => (
                      <option key={v} value={v}>{v === 0 ? "Manual Only" : `${v}s`}</option>
                    ))}
                  </select>
                ),
              },
              {
                label: "High Contrast / Large Text Mode",
                desc: "Enhances typography size and contrast for increased readability.",
                el: (
                  <button
                    type="button"
                    className={`btn ${largeText ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => setLargeText(!largeText)}
                  >
                    {largeText ? "Enabled" : "Disabled"}
                  </button>
                ),
              },
            ].map(s => (
              <div key={s.label} className="setting-row">
                <div className="setting-info">
                  <h4>{s.label}</h4>
                  <p>{s.desc}</p>
                </div>
                {s.el}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "chart" && (
        <div className="card">
          <div className="card-header">
            <h3>Chart Display Options</h3>
          </div>
          <div className="card-body">
            {[
              { key: "showArea",     label: "Shaded Area Fill",        desc: "Renders smooth gradient area under the CGM curve." },
              { key: "showForecast", label: "30-Min ODE Forecast",      desc: "Displays the mechanistic forecast projection trajectory in purple dashed style." },
              { key: "showZones",    label: "Target Glucose Zone",     desc: "Highlights the clinical consensus 70–180 mg/dL target band." },
            ].map(s => (
              <div key={s.key} className="setting-row">
                <div className="setting-info">
                  <h4>{s.label}</h4>
                  <p>{s.desc}</p>
                </div>
                <button
                  type="button"
                  className={`btn ${chartPrefs[s.key] ? "btn-primary" : "btn-secondary"}`}
                  onClick={() => setChartPrefs((p: any) => ({ ...p, [s.key]: !p[s.key] }))}
                >
                  {chartPrefs[s.key] ? "Shown" : "Hidden"}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "research" && (
        <div className="grid-2">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {phases.map(phase => (
              <div
                key={phase.id}
                className="card"
                style={{
                  cursor: "pointer",
                  borderColor: selectedPhase?.id === phase.id ? "var(--emerald-600)" : undefined,
                  background: selectedPhase?.id === phase.id ? "var(--emerald-50)" : undefined,
                }}
                onClick={() => setSelectedPhase(phase)}
              >
                <div className="card-body" style={{ padding: "12px 16px" }}>
                  <div style={{ fontSize: 13.5, fontWeight: 700, color: selectedPhase?.id === phase.id ? "var(--emerald-800)" : "var(--text-main)" }}>
                    {phase.title}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 3 }}>
                    {phase.description}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div>
            {selectedPhase ? (
              <div className="card">
                <div className="card-header">
                  <h3>{selectedPhase.title}</h3>
                  <Badge label={selectedPhase.data_origin} type="neutral" />
                </div>
                <div className="card-body">
                  {selectedPhase.plot_available ? (
                    <img
                      src={`/api/experiments/${selectedPhase.plot}/image`}
                      alt={selectedPhase.title}
                      style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border-color)", marginBottom: 12 }}
                    />
                  ) : (
                    <div className="empty-state" style={{ padding: 18 }}>
                      <p>Scientific plot artifact generating...</p>
                    </div>
                  )}
                  {loadingReport ? (
                    <Spinner />
                  ) : (
                    report && (
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, background: "var(--bg-secondary)", padding: 12, borderRadius: 8, maxHeight: 260, overflowY: "auto", whiteSpace: "pre-wrap" }}>
                        {report}
                      </div>
                    )
                  )}
                </div>
              </div>
            ) : (
              <div className="card">
                <div className="empty-state">
                  <p>Select a research phase on the left to inspect reports and artifact plots.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "privacy" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="info-callout">
            <ShieldCheck size={22} className="info-callout-icon" />
            <div className="info-callout-text">
              <h4>Local-First Execution Architecture</h4>
              <p>100% of calculations, model simulations, SQLite storage, and telemetry datasets remain strictly on your local computer.</p>
            </div>
          </div>
          <div className="card">
            <div className="card-header">
              <h3>Data Persistence Boundaries</h3>
            </div>
            <div className="card-body">
              <ul style={{ paddingLeft: 18, display: "flex", flexDirection: "column", gap: 8, fontSize: 13, color: "var(--text-body)" }}>
                <li><strong>Virtual patient profiles:</strong> Local SQLite database (<code>data/processed/patients.db</code>)</li>
                <li><strong>User-entered meals:</strong> Local JSON storage (<code>data/processed/user_meals.json</code>)</li>
                <li><strong>Interface preferences:</strong> Browser localStorage</li>
                <li><strong>External network calls:</strong> 0 external connections. Completely offline capable.</li>
              </ul>
            </div>
          </div>
          <div className="card">
            <div className="card-body" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
              <div>
                <h4 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-main)" }}>Reset All Local Preferences</h4>
                <p style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>Clears custom patient aliases, chart settings, and localStorage state.</p>
              </div>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => { if (window.confirm("Reset all local settings and aliases to default?")) onResetPreferences(); }}
              >
                Reset to Defaults
              </button>
            </div>
          </div>
        </div>
      )}

      {health && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-header">
            <h3><Activity size={15} />Backend Diagnostic Health</h3>
            <Badge label="FastAPI" type="target" />
          </div>
          <div className="card-body">
            <div className="metrics-row" style={{ marginBottom: 0 }}>
              {[
                { label: "API Status", val: health.api, Icon: ShieldCheck, cls: "success" },
                { label: "Dataset Ready", val: health.dataset_ready ? "Yes" : "No", Icon: Activity, cls: "success" },
                { label: "Patient Files", val: `${health.patient_count}`, Icon: User, cls: "neutral" },
                { label: "Mechanistic ODE", val: health.mechanistic_model, Icon: Zap, cls: "teal" },
              ].map(t => (
                <div key={t.label} className="metric-card">
                  <div className={`metric-icon-wrap ${t.cls}`}>
                    <t.Icon size={16} />
                  </div>
                  <div className="metric-info">
                    <div className="metric-label">{t.label}</div>
                    <div className="metric-value" style={{ fontSize: 18 }}>{t.val}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Root App Component ────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("twin");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [selectedId, setSelectedId] = useState("synthetic_000");
  const [patients, setPatients] = useState<PatientListItem[]>([]);

  const [profileNames, setProfileNamesState] = useState<Record<string, string>>(() => {
    try { return JSON.parse(localStorage.getItem("t1d_profile_names") || "{}"); } catch { return {}; }
  });
  const [defaultWindowHours, setDefaultWindowHoursState] = useState<number>(() => {
    try { return Number(localStorage.getItem(LS_DEFAULT_WINDOW) || "24"); } catch { return 24; }
  });
  const [refreshInterval, setRefreshIntervalState] = useState<number>(() => {
    try { const s = localStorage.getItem(LS_REFRESH_INTERVAL); return s !== null ? Number(s) : 0; } catch { return 0; }
  });
  const [chartPrefs, setChartPrefs] = useState<{ showArea: boolean; showForecast: boolean; showZones: boolean }>(() => {
    try {
      return JSON.parse(localStorage.getItem(LS_CHART_PREFS) || "null") || { showArea: true, showForecast: true, showZones: true };
    } catch {
      return { showArea: true, showForecast: true, showZones: true };
    }
  });
  const [largeText, setLargeText] = useState<boolean>(() => {
    try { return localStorage.getItem(LS_LARGE_TEXT) === "true"; } catch { return false; }
  });

  useEffect(() => {
    document.body.classList.toggle("large-text-mode", largeText);
    try { localStorage.setItem(LS_LARGE_TEXT, String(largeText)); } catch {}
  }, [largeText]);

  useEffect(() => {
    try { localStorage.setItem(LS_CHART_PREFS, JSON.stringify(chartPrefs)); } catch {}
  }, [chartPrefs]);

  const setProfileName = useCallback((id: string, name: string) => {
    setProfileNamesState(prev => {
      const next = { ...prev, [id]: name };
      try { localStorage.setItem("t1d_profile_names", JSON.stringify(next)); } catch {}
      return next;
    });
  }, []);

  const setDefaultWindowHours = useCallback((h: number) => {
    setDefaultWindowHoursState(h);
    try { localStorage.setItem(LS_DEFAULT_WINDOW, String(h)); } catch {}
  }, []);

  const setRefreshInterval = useCallback((s: number) => {
    setRefreshIntervalState(s);
    try { localStorage.setItem(LS_REFRESH_INTERVAL, String(s)); } catch {}
  }, []);

  const handleResetPreferences = useCallback(() => {
    ["t1d_profile_names", LS_DEFAULT_WINDOW, LS_REFRESH_INTERVAL, LS_CHART_PREFS, LS_LARGE_TEXT].forEach(k => {
      try { localStorage.removeItem(k); } catch {}
    });
    setProfileNamesState({});
    setDefaultWindowHoursState(24);
    setRefreshIntervalState(0);
    setChartPrefs({ showArea: true, showForecast: true, showZones: true });
    setLargeText(false);
  }, []);

  useEffect(() => {
    axios.get("/api/health")
      .then(r => setHealth(r.data))
      .catch(() => setHealth(null));
    axios.get("/api/patients")
      .then(r => {
        setPatients(r.data.patients);
        if (r.data.patients.length > 0 && !selectedId) setSelectedId(r.data.patients[0].id);
      })
      .catch(() => {});
  }, []);

  const renderPage = () => {
    switch (page) {
      case "overview":
        return (
          <ErrorBoundary fallbackTitle="Overview Dashboard Error">
            <OverviewPage
              selectedId={selectedId}
              setSelectedId={setSelectedId}
              profileNames={profileNames}
              setProfileName={setProfileName}
              defaultWindowHours={defaultWindowHours}
              refreshInterval={refreshInterval}
              chartPrefs={chartPrefs}
            />
          </ErrorBoundary>
        );
      case "twin":
        return (
          <ErrorBoundary fallbackTitle="Digital Twin Engine Error">
            <DigitalTwinPage
              selectedId={selectedId}
              setSelectedId={setSelectedId}
              chartPrefs={chartPrefs}
              defaultWindowHours={defaultWindowHours}
            />
          </ErrorBoundary>
        );
      case "whatif":
        return (
          <ErrorBoundary fallbackTitle="What-If Simulation Lab Error">
            <WhatIfPage />
          </ErrorBoundary>
        );
      case "comparison":
        return (
          <ErrorBoundary fallbackTitle="Model Benchmark Error">
            <ModelComparisonPage />
          </ErrorBoundary>
        );
      case "history":
        return (
          <ErrorBoundary fallbackTitle="Clinical History Error">
            <HistoryPage selectedId={selectedId} patients={patients} />
          </ErrorBoundary>
        );
      case "settings":
        return (
          <ErrorBoundary fallbackTitle="Platform Settings Error">
            <SettingsPage
              profileNames={profileNames}
              setProfileName={setProfileName}
              defaultWindowHours={defaultWindowHours}
              setDefaultWindowHours={setDefaultWindowHours}
              refreshInterval={refreshInterval}
              setRefreshInterval={setRefreshInterval}
              chartPrefs={chartPrefs}
              setChartPrefs={setChartPrefs}
              largeText={largeText}
              setLargeText={setLargeText}
              onResetPreferences={handleResetPreferences}
              health={health}
            />
          </ErrorBoundary>
        );
      default:
        return null;
    }
  };

  return (
    <div className="app-shell">
      {/* Permanent Left Sidebar (Only Navigation) */}
      <Sidebar
        page={page}
        setPage={setPage}
        collapsed={sidebarCollapsed}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      {/* Compact Top Header */}
      <CompactHeader
        page={page}
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
        health={health}
      />

      {/* Main Content Area */}
      <main className={`main-area ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <div className="page-content">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
