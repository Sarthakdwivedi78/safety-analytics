"""Streamlit front end for the Nightshift Lab capstone."""
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_model import (ARCHIVE_URL, CAMERA, FEATURES, FIELDS, LIGHT, LUX, NOISE, SOURCE_URL, UNITS,
                        cohort_available, cohort_rows, dark_rows, demo, normalize, official_archive)
from governor import PRESETS, STPA, evaluate
from research_model import InsufficientEvidence, fit_research

st.set_page_config(page_title="Nightshift Lab", page_icon="◈", layout="wide")
st.sidebar.title("◈ Nightshift Lab")
st.sidebar.caption("AUTONOMOUS SYSTEMS · CAPSTONE 02")
page = st.sidebar.radio("Research workspace", ["Overview", "Data Explorer", "Modeling & Results",
                                               "STPA Safety", "Braking Simulator", "Report & Downloads"])
source = st.sidebar.selectbox("Active dataset", ["Synthetic demo", "NHTSA ADS archive", "Uploaded telemetry"])
target = st.sidebar.slider("Calibration risk target (%)", 1, 25, 5) / 100
upload = None
if source == "Uploaded telemetry":
    upload = st.sidebar.file_uploader("Telemetry CSV", type=["csv"])


@st.cache_data(show_spinner=False)
def built_in(which):
    return demo() if which == "Synthetic demo" else official_archive()


def read_upload(file):
    if file is None:
        return None
    frame = pd.read_csv(file, dtype="string")
    if frame.empty or len(frame) > 100_000:
        raise ValueError("The CSV must contain between 1 and 100,000 data rows.")
    return frame


def safe_export(frame, name):
    frame = frame.copy()
    for col in frame.select_dtypes(include=["object", "string"]):
        frame[col] = frame[col].map(lambda x: "'" + x if isinstance(x, str) and
                                   x.lstrip().startswith(("=", "+", "-", "@")) else x)
    frame.insert(0, "data_source", source)
    st.download_button(name, frame.to_csv(index=False).encode(), file_name=name.lower().replace(" ", "-")+".csv",
                       mime="text/csv", key=name)


@st.cache_resource(show_spinner=False, max_entries=8)
def demo_model(risk_target):
    return fit_research(normalize(demo())[0], risk_target)


def analyze(data, show_errors=True):
    if data is None or source == "NHTSA ADS archive":
        if show_errors:
            st.info("Choose the synthetic demo or upload labeled telemetry with independent session IDs.")
        return None
    try:
        with st.spinner("Fitting models and evaluating held-out sessions…"):
            if source == "Synthetic demo":
                return demo_model(target)
            return fit_research(data, target)
    except InsufficientEvidence as exc:
        if show_errors:
            st.warning(f"Insufficient evidence. {exc}")
        return None


try:
    raw = read_upload(upload) if source == "Uploaded telemetry" else built_in(source)
except (ValueError, UnicodeError, pd.errors.ParserError) as error:
    st.error(f"Unable to read CSV. {error}")
    raw = None

mapping = None
if source == "Uploaded telemetry" and raw is not None:
    mapping = {}
    with st.sidebar.expander("Map CSV columns"):
        options = [None, *raw.columns]
        for field in FIELDS:
            default = field if field in raw else "Lighting" if field == LIGHT and "Lighting" in raw else None
            mapping[field] = st.selectbox(field, options, index=options.index(default),
                                          format_func=lambda x: "Unavailable" if x is None else x)
data, invalid = normalize(raw, mapping) if raw is not None else (None, {})
available = data is not None and cohort_available(data)
summary = {
    "Rows": len(data) if data is not None else 0,
    "Dark / no lights": int(dark_rows(data).sum()) if data is not None else 0,
    "Exact cohort": int(cohort_rows(data).sum()) if available else None,
    "Known outcomes": int(data.outcome.notna().sum()) if data is not None else 0,
}


def banner():
    if source == "Synthetic demo":
        st.info("SYNTHETIC DEMO · 4,200 simulated observations across 150 sessions. Results describe the simulation.")
    elif source == "NHTSA ADS archive":
        st.info("OBSERVED INCIDENT CONTEXT · Latest version of each archived report ID. The archive lacks the paired telemetry and outcome denominator for model fitting.")
    else:
        st.info("USER-SUPPLIED TELEMETRY · Analysis uses the selected mappings. Uploads are processed by the app for this session.")


def metrics(items):
    for cell, (name, value) in zip(st.columns(len(items)), items.items()):
        if value is None:
            value = "Unavailable"
        elif isinstance(value, int):
            value = f"{value:,}"
        cell.metric(name, value)


def fmt(x):
    return f"{x:.2%}" if np.isfinite(x) else "Unavailable"


def bounds(ci):
    return " to ".join(fmt(x) for x in ci) if np.isfinite(ci).all() else "Unavailable"


def stpa_table():
    return pd.DataFrame(STPA, columns=["ID", "Unsafe action", "Trace", "Causal scenario",
                                      "Safety constraint", "Simulator preset"])


def chart(data, x, y, color=None, points=False):
    if len(data) > 4_500:
        if points:
            data = data.sample(4_500, random_state=24021)
        elif color:
            data = pd.concat([part.iloc[np.linspace(0, len(part)-1, min(len(part), 1200), dtype=int)]
                              for _, part in data.groupby(color, sort=False)], ignore_index=True)
    base = alt.Chart(data)
    base = base.mark_circle(size=42, opacity=.65) if points else base.mark_line(point=True)
    axes = {"x": alt.X(x, type="quantitative"), "y": alt.Y(y, type="quantitative")}
    if color:
        axes["color"] = alt.Color(color, type="nominal")
    st.altair_chart(base.encode(**axes).properties(height=280).interactive(), width="stretch")


if page == "Overview":
    st.caption("PROJECT 02  /  AUTONOMOUS SYSTEMS SAFETY")
    st.title("Nighttime Vision Sensor Attenuation")
    st.subheader("Dense urban platoon corridors")
    st.markdown("When does degraded camera information become unsafe for automated emergency braking?")
    banner()
    metrics(summary)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Research hypothesis")
        st.write("Low light and sensor noise reduce camera detection confidence. A late or missed brake request can leave a platoon follower too little stopping distance.")
    with right:
        st.markdown("#### Safety constraint")
        st.write("When the camera confidence gate fails, usable radar takes primary braking authority. If radar is also unavailable, the supervisor enters a reduced or minimum-risk mode.")
    st.markdown("#### Research workflow")
    st.write("Audit the telemetry → model attenuation and adverse outcomes → calibrate a camera gate → evaluate on held-out sessions → trace the simulated braking decision.")

elif page == "Data Explorer":
    st.title("Data Explorer")
    banner()
    metrics(summary)
    st.markdown(f"[NHTSA crash reporting archive]({SOURCE_URL}) · [Official ADS CSV]({ARCHIVE_URL})")
    if source == "NHTSA ADS archive":
        st.write("Archived snapshot through June 2025. From 2,295 source rows, retain the highest version for each of 1,899 report IDs. Separate report IDs can describe the same incident. The archive’s Dark - Not Lighted maps to Dark-No-Lights.")
        st.bar_chart(data[LIGHT].fillna("Unknown").value_counts(), horizontal=True)
    if data is not None:
        fields = pd.DataFrame({"field": FIELDS, "unit or meaning": UNITS,
                               "missing": [int(data[x].isna().sum()) for x in FIELDS],
                               "invalid numeric": [invalid.get(x, 0) for x in FIELDS]})
        st.dataframe(fields, hide_index=True)
        st.dataframe(data.head(16), hide_index=True)
        safe_export(fields, "Field coverage")
        safe_export(pd.DataFrame(summary.items(), columns=["metric", "value"]), "Dataset summary")
    st.download_button("Download telemetry template", demo().head(3).assign(data_source="Synthetic example").to_csv(index=False).encode(),
                       "synthetic-telemetry-template.csv", "text/csv")

elif page == "Modeling & Results":
    st.title("Modeling & Results")
    banner()
    result = analyze(data)
    if result:
        st.subheader("Independent session evaluation")
        st.caption(f"{result['exact']:,} exact cohort rows. {result['excluded']:,} excluded for missing session, features, or outcome.")
        st.dataframe(result["sizes"], hide_index=True)
        st.dataframe(result["metrics"].round(4), hide_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown("#### ROC curves")
            chart(result["roc"], "FPR", "TPR", "model")
        with right:
            st.markdown("#### Probability reliability")
            chart(result["reliability"], "mean predicted", "observed fraction", "model", points=True)
            st.caption("Each point is a probability bin from the held-out set.")
        st.subheader("Camera confidence gate")
        st.caption(f"Calibration requires at least 40 retained observations and 10 sessions, and a one-sided Wilson upper bound at or below {target:.0%}. This is a screening rule.")
        chart(result["candidates"], "threshold", "upper_95_one_sided")
        if result["gate"] is None:
            st.warning("No camera gate meets the calibration support and risk criteria. Inspect the candidate table before trying a different target.")
        else:
            g = result["gate_result"]
            metrics({"Candidate": f"{result['gate']:.2f}", "Held-out adverse rate": fmt(g["retained_rate"]),
                     "Adverse cases retained": f"{g['missed_adverse']} / {g['total_adverse']}"})
            st.write(f"Session bootstrap interval for retained adverse rate: {bounds(g['retained_rate_cluster_ci'])}. "
                     f"False-negative rate interval: {bounds(g['false_negative_rate_cluster_ci'])}.")
            rr = g["relative_risk"]
            if np.isfinite(rr):
                lo, hi = g["relative_risk_cluster_ci"]
                rr_interval = f"{lo:.2f}–{hi:.2f}×" if np.isfinite([lo, hi]).all() else "unavailable"
                st.write(f"Below / retained relative risk: {rr:.2f}×. Session interval: {rr_interval}.")
            else:
                st.write("Relative risk is unavailable when the retained group has no observations or no adverse events.")
            st.dataframe(result["gate_rows"].round(4), hide_index=True)
        with st.expander("Candidate table"):
            st.dataframe(result["candidates"].round(4), hide_index=True)
        st.subheader("Sensor fusion and attenuation")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Two-component PCA")
            chart(result["projection"], "PC1", "PC2", "Adverse", points=True)
            st.caption("Explained variance: " + " / ".join(fmt(v) for v in result["variance"]))
            st.dataframe(result["loadings"].round(4), hide_index=True)
        with right:
            st.markdown("#### Low light and sensor noise")
            fit = result["attenuation"]
            if fit:
                chart(fit["points"], CAMERA, "predicted", points=True)
                st.caption(f"Held-out RMSE {fit['rmse']:.3f}. Fit on {fit['train']:,} observations and evaluated on {fit['test']:,}.")
                st.write(pd.Series(fit["coefficients"], name="Coefficient").round(4))
            else:
                st.info("The comparison cohort needs usable illumination and noise in lit and unlit sessions.")
        with st.expander("Takeover latency and logistic coefficients"):
            scale = result["latency"]
            if scale:
                st.write(f"Training latency range {scale['min']:.3f}–{scale['max']:.3f} s. "
                         "Held-out values are clipped to [0, 1]. Constant training columns map to 0.")
                st.dataframe(scale["table"].head(15), hide_index=True)
            st.caption(f"Logistic intercept: {result['intercept']:.4f}. Latency is analyzed separately from the pre-brake predictors.")
            st.dataframe(result["effects"].round(4), hide_index=True)

elif page == "STPA Safety":
    st.title("STPA Safety Analysis")
    st.subheader("Issue Automated Platoon Emergency Brake")
    st.code("Sensor feedback → Braking supervisor → Brake actuator → Platoon state\n"
            "      ↑_________________________________________________________|", language=None)
    st.write("L1: collision injury or damage. L2: unnecessary deceleration and platoon disruption.")
    st.write("H1: insufficient braking. H2: braking without an obstacle. H3: release while the hazard remains.")
    st.dataframe(stpa_table(), hide_index=True)
    st.caption("Each named verification case can be run from the Braking Simulator.")

elif page == "Braking Simulator":
    st.title("Braking Simulator")
    st.info("Research simulation only. TTC uses a stationary obstacle. Demand and actuator deadlines are illustrative.")
    preset = st.selectbox("Scenario preset", list(PRESETS))
    original, expected_authority, expected_action = PRESETS[preset]
    custom = st.checkbox("Adjust scenario inputs")
    scenario = original
    if custom:
        a, b, c = st.columns(3)
        updates = {}
        for col, sensor in zip((a, b, c), ("camera", "radar", "lidar")):
            with col:
                st.markdown(f"#### {sensor.title()}")
                score = getattr(original, sensor)
                updates[sensor+"_healthy"] = st.checkbox("Healthy", getattr(original, sensor+"_healthy"), key=preset+sensor+"health")
                present = st.checkbox("Reading available", score is not None, key=preset+sensor+"present")
                updates[sensor] = st.slider("Confidence", 0., 1., score or 0., .01, key=preset+sensor+"score") if present else None
        a, b, c = st.columns(3)
        updates["camera_gate"] = a.slider("Camera threshold", .1, .9, original.camera_gate, .01)
        updates["speed"] = b.slider("Speed (km/h)", 0, 100, original.speed)
        updates["gap"] = c.slider("Gap (m)", 2, 100, original.gap)
        updates["obstacle"] = a.checkbox("Obstacle present", original.obstacle)
        updates["previous_request"] = b.checkbox("Brake already requested", original.previous_request)
        updates["feedback_delay"] = c.slider("Feedback delay (s)", 0., 1., original.feedback_delay, .01)
        scenario = replace(original, **updates)
    evaluated = evaluate(scenario)
    metrics({"Authority": evaluated["authority"], "Time to obstacle": f"{evaluated['ttc']:.2f} s" if evaluated["ttc"] is not None else "—"})
    st.subheader(evaluated["action"])
    if not custom:
        match = (evaluated["authority"], evaluated["action"]) == (expected_authority, expected_action)
        (st.success if match else st.error)("Preset verification " + ("passed." if match else "failed."))
    st.dataframe(pd.DataFrame([{"gate": name, "status": "Pass" if evaluated[key] else "Fail",
                                "condition": condition} for name, key, condition in [
        ("Camera", "camera", f"Healthy reading ≥ {scenario.camera_gate:.2f}"),
        ("Radar", "radar", "Healthy reading ≥ 0.50"),
        ("Lidar", "lidar", "Healthy reading ≥ 0.50")]]), hide_index=True)
    with st.expander("Experimental adverse outcome score"):
        st.write("The classifier was fitted within dark, camera-blind observations. Use it here only for that scenario. The simulator determines authority separately.")
        l, n = st.columns(2)
        lux = l.number_input("Illumination (lux)", min_value=0., value=2.)
        noise = n.slider("Normalized sensor noise", 0., 1., .60)
        if st.button("Score dark, camera-blind scenario"):
            sensor_values = [scenario.camera, scenario.radar, scenario.lidar]
            healthy = scenario.camera_healthy and scenario.radar_healthy and scenario.lidar_healthy
            if not healthy or any(v is None for v in sensor_values):
                st.warning("A score requires valid readings from all three sensors.")
            else:
                result = analyze(data)
                if result:
                    row = pd.DataFrame([[*sensor_values, lux, noise, scenario.speed, scenario.gap]], columns=FEATURES)
                    risk = result["model"].predict_proba(row)[0, 1]
                    st.metric("Estimated adverse outcome probability", fmt(risk))
                    st.caption(source + " · Exploratory model output")

else:
    st.title("Report & Downloads")
    banner()
    result = analyze(data, show_errors=False)
    report = (Path(__file__).parent / "CAPSTONE.md").read_text(encoding="utf-8")
    if result:
        fusion = result["metrics"].set_index("model").loc["Fusion + road context"]
        report += (f"\n\n## Current run\n\nDataset: {source}. The held-out fusion model produced "
                   f"ROC AUC {fusion['ROC AUC']:.3f}, average precision {fusion['Average precision']:.3f}, "
                   f"and Brier score {fusion['Brier score']:.3f}. ")
        if result["gate"] is None:
            report += "No candidate met the calibration risk and support criteria."
        else:
            g = result["gate_result"]
            report += (f"Calibration candidate {result['gate']:.2f} at target {target:.1%}. Held-out retained adverse rate "
                       f"{fmt(g['retained_rate'])} (session interval {bounds(g['retained_rate_cluster_ci'])}).")
        for name, key in [("Model metrics", "metrics"), ("Camera candidates", "candidates"),
                          ("PCA loadings", "loadings"), ("ROC curve", "roc"),
                          ("Reliability bins", "reliability"), ("Partition sizes", "sizes")]:
            safe_export(result[key], name)
        if result["gate"] is not None:
            safe_export(result["gate_rows"], "Held-out gate groups")
    elif source == "NHTSA ADS archive":
        report += f"\n\n## Current run\n\nObserved incident context: {summary['Dark / no lights']} unlit reports among {summary['Rows']} report IDs. The exact camera-blind cohort is unavailable."
    st.download_button("Download capstone report", report.encode(), "nightshift-capstone.md", "text/markdown")
    st.markdown(report)
