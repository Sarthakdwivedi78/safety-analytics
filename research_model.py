"""Grouped training, calibration, and independent evaluation of research models."""
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_model import CAMERA, FEATURES, GROUP, LATENCY, LIGHT, LUX, NOISE, SENSORS, cohort_rows


class InsufficientEvidence(ValueError):
    """The input does not support a defensible grouped analysis."""


@dataclass
class Split:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def grouped_split(data):
    if data[GROUP].nunique() < 15 or data["outcome"].value_counts().reindex([0, 1], fill_value=0).min() < 15:
        raise InsufficientEvidence("Need at least 15 independent sessions and 15 examples of each outcome.")
    if data["outcome"].nunique() != 2 or data.loc[data.outcome.eq(1), GROUP].nunique() < 5:
        raise InsufficientEvidence("Adverse and non-adverse examples must be present across independent sessions.")
    for seed in range(24021, 24029):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [idx for _, idx in splitter.split(data, data.outcome, data[GROUP])]
        sets = Split(data.iloc[np.concatenate(folds[:3])], data.iloc[folds[3]], data.iloc[folds[4]])
        if all(x["outcome"].value_counts().reindex([0, 1], fill_value=0).min() >= 3
               for x in (sets.train, sets.calibration, sets.test)):
            return sets
    raise InsufficientEvidence("The session groups could not be split with sufficient outcomes in all partitions.")


def wilson(events, n, confidence=.95, one_sided=False):
    if not n:
        return (np.nan, np.nan)
    p = events / n
    z = NormalDist().inv_cdf(confidence if one_sided else (1+confidence)/2)
    d = 1+z*z/n
    m = (p+z*z/(2*n))/d
    w = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0., m-w), min(1., m+w)


def threshold_table(frame, thresholds):
    rows = []
    for threshold in thresholds:
        retained = frame.loc[frame[CAMERA].ge(threshold)]
        n, events = len(retained), int(retained.outcome.sum())
        rows.append({"threshold": threshold, "retained": n, "sessions": retained[GROUP].nunique(),
                     "events": events, "event_rate": events/n if n else np.nan,
                     "upper_95_one_sided": wilson(events, n, one_sided=True)[1]})
    return pd.DataFrame(rows)


def select_gate(calibration, target):
    candidates = threshold_table(calibration, np.linspace(.10, .90, 41).round(2))
    ok = candidates[(candidates.retained >= 40) & (candidates.sessions >= 10) &
                    (candidates.upper_95_one_sided <= target)]
    return (float(ok.iloc[0].threshold) if len(ok) else None), candidates


def gate_metrics(test, gate, draws=250):
    below = test[CAMERA].lt(gate).to_numpy()
    y = test.outcome.to_numpy()
    rows = []
    for name, mask in [("Below threshold", below), ("At or above threshold", ~below)]:
        events, n = int(y[mask].sum()), int(mask.sum())
        lo, hi = wilson(events, n)
        rows.append({"group": name, "observations": n, "adverse": events,
                     "rate": events/n if n else np.nan, "wilson_low": lo, "wilson_high": hi})
    group_df = pd.DataFrame(rows)
    cluster_indices = list(test.groupby(GROUP, sort=True).indices.values())
    rng = np.random.default_rng(477)
    boot = []
    for _ in range(draws):
        idx = np.concatenate([cluster_indices[i] for i in rng.integers(0, len(cluster_indices), len(cluster_indices))])
        yb, lb = y[idx], below[idx]
        high = yb[~lb].mean() if (~lb).any() else np.nan
        low = yb[lb].mean() if lb.any() else np.nan
        boot.append([high, yb[~lb].sum()/yb.sum() if yb.sum() else np.nan,
                     low/high if high > 0 else np.nan])
    bounds = []
    for col in np.array(boot).T:
        valid = col[np.isfinite(col)]
        # A degenerate bootstrap cannot establish an informative uncertainty interval.
        bounds.append(tuple(np.quantile(valid, [.025, .975]))
                      if len(valid) >= .9*draws and np.ptp(valid) > 0 else (np.nan, np.nan))
    low, high = group_df.rate
    return group_df, {
        "missed_adverse": int(y[~below].sum()), "total_adverse": int(y.sum()),
        "handover_fraction": float(below.mean()), "retained_rate": high,
        "retained_rate_cluster_ci": bounds[0], "false_negative_rate_cluster_ci": bounds[1],
        "relative_risk": low/high if high > 0 else np.nan, "relative_risk_cluster_ci": bounds[2],
    }


def attenuation(data, sessions):
    comparison = data[data[LIGHT].isin(["Dark-No-Lights", "Dark - Lighted", "Daylight", "Dawn / Dusk"])]
    comparison = comparison.dropna(subset=[GROUP, LUX, NOISE, CAMERA]).copy()
    train, test = [comparison[comparison[GROUP].isin(groups)] for groups in sessions]
    def design(frame):
        l = frame[LUX].to_numpy()/10
        n = frame[NOISE].to_numpy()
        return np.column_stack([l, n, l*n])
    def both_lighting(frame):
        dark = frame[LIGHT].eq("Dark-No-Lights")
        return dark.any() and (~dark).any()
    if len(train) < 40 or len(test) < 15 or not both_lighting(train) or not both_lighting(test):
        return None
    x = design(train)
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(train)), x])) < 4:
        return None
    model = LinearRegression().fit(x, train[CAMERA])
    pred = model.predict(design(test))
    return {"coefficients": dict(zip(["intercept", "lux_per_10", "noise", "lux_noise"],
                                     [model.intercept_, *model.coef_])),
            "rmse": float(np.sqrt(np.mean((pred-test[CAMERA])**2))), "train": len(train),
            "test": len(test), "points": test[[LUX, NOISE, CAMERA]].assign(predicted=pred)}


def fit_research(data, target=.05):
    if not 0 < target <= .5:
        raise ValueError("Risk target must be between 0 and 0.50.")
    exact = data.loc[cohort_rows(data)]
    complete = exact.dropna(subset=[GROUP, *FEATURES, "outcome"])
    if complete.empty:
        raise InsufficientEvidence("No complete, labeled telemetry rows in the exact cohort.")
    splits = grouped_split(complete)
    train, calibration, test = splits.train, splits.calibration, splits.test
    if (train[SENSORS].std() <= 1e-9).any():
        raise InsufficientEvidence("Three nonconstant confidence channels are needed for PCA.")
    sensor_pipe = Pipeline([("standardize", StandardScaler()), ("pca", PCA(n_components=2))])
    context = [LUX, NOISE, *FEATURES[-2:]]
    transform = ColumnTransformer([("sensor", sensor_pipe, SENSORS),
                                   ("context", StandardScaler(), context)])
    fusion = Pipeline([("transform", transform), ("regression", LogisticRegression(max_iter=1000))])
    baseline = Pipeline([("scale", StandardScaler()), ("regression", LogisticRegression(max_iter=1000))])
    fusion.fit(train[FEATURES], train.outcome)
    baseline.fit(train[[CAMERA]], train.outcome)
    pc = fusion.named_steps["transform"].named_transformers_["sensor"]
    pca = pc.named_steps["pca"]
    if pca.explained_variance_[1] <= 1e-9:
        raise InsufficientEvidence("The sensor channels do not span two independent directions.")
    projection = pd.DataFrame(pc.transform(test[SENSORS]), columns=["PC1", "PC2"], index=test.index)
    projection["Adverse"] = test.outcome.map({0.: "No", 1.: "Yes"})
    predictions = {"Prevalence": np.full(len(test), train.outcome.mean()),
                   "Camera only": baseline.predict_proba(test[[CAMERA]])[:, 1],
                   "Fusion + road context": fusion.predict_proba(test[FEATURES])[:, 1]}
    metrics, curves, calibration_bins = [], [], []
    for name, probabilities in predictions.items():
        metrics.append({"model": name, "ROC AUC": roc_auc_score(test.outcome, probabilities),
                        "Average precision": average_precision_score(test.outcome, probabilities),
                        "Brier score": brier_score_loss(test.outcome, probabilities)})
        fpr, tpr, _ = roc_curve(test.outcome, probabilities)
        curves.extend({"model": name, "FPR": x, "TPR": y} for x, y in zip(fpr, tpr))
        frame = pd.DataFrame({"score": probabilities, "outcome": test.outcome.to_numpy()})
        for _, group in frame.groupby(pd.cut(frame.score, np.linspace(0, 1, 11), include_lowest=True), observed=True):
            calibration_bins.append({"model": name, "mean predicted": group.score.mean(),
                                     "observed fraction": group.outcome.mean(), "n": len(group)})
    gate, candidates = select_gate(calibration, target)
    gate_rows, gate_result = gate_metrics(test, gate) if gate is not None else (pd.DataFrame(), None)
    train_latency = train[[LATENCY]].dropna()
    test_latency = test[[LATENCY]].dropna()
    scale = None
    if not train_latency.empty:
        scaler = MinMaxScaler(clip=True).fit(train_latency)
        constant = scaler.data_range_[0] == 0
        scaled = (np.zeros(len(test_latency)) if constant else
                  scaler.transform(test_latency).ravel() if len(test_latency) else np.empty(0))
        scale = {"min": scaler.data_min_[0], "max": scaler.data_max_[0], "constant": bool(constant),
                 "train_n": len(train_latency),
                 "table": pd.DataFrame({"seconds": test_latency[LATENCY], "scaled": scaled})}
    coefficients = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=SENSORS)
    coefficients.index.name = "sensor"
    coefficients = coefficients.reset_index()
    sizes = pd.DataFrame([{"partition": name, "rows": len(part), "sessions": part[GROUP].nunique(),
                           "adverse": int(part.outcome.sum())}
                          for name, part in [("Training", train), ("Calibration", calibration), ("Held-out", test)]])
    logistic = fusion.named_steps["regression"]
    effects = pd.DataFrame({"standardized input": ["PC1", "PC2", *context],
                            "log-odds coefficient": logistic.coef_[0]})
    return {"exact": len(exact), "excluded": len(exact)-len(complete), "sizes": sizes,
            "metrics": pd.DataFrame(metrics), "roc": pd.DataFrame(curves),
            "reliability": pd.DataFrame(calibration_bins), "projection": projection,
            "loadings": coefficients, "variance": pca.explained_variance_ratio_,
            "gate": gate, "gate_rows": gate_rows, "gate_result": gate_result, "candidates": candidates,
            "latency": scale, "effects": effects, "intercept": float(logistic.intercept_[0]),
            "attenuation": attenuation(data, [train[GROUP].unique(), test[GROUP].unique()]),
            "model": fusion}
