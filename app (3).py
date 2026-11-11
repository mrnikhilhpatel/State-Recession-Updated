"""
State Economic Resilience Dashboard (v1 draft)
2008 housing crisis vs. 2020 COVID-19 recession, 50 states + DC.

Run:   streamlit run app.py
Data:  data/fred_state_panel_long.csv  (columns: state, indicator, date, value, ...)
       If the file is missing, the app asks you to upload it.
Everything (features, resilience score, clusters, stability tests) is computed on load.
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

st.set_page_config(page_title="State Economic Resilience", page_icon="📉", layout="wide")

# ----------------------------------------------------------------------------
# SETTINGS (edit here)
# ----------------------------------------------------------------------------
DATA_PATHS = ["data/fred_state_panel_long.csv", "fred_state_panel_long.csv"]

RECESSIONS = {
    "2008": {"label": "2008 Housing Crisis",
             "nber_start": "2007-12-01", "nber_end": "2009-06-01",
             "baseline_months": 12, "peak_search_months": 48,
             "recovery_months": 24, "lfpr_months": 24,
             "housing_window": ("2004-01-01", "2012-12-31")},
    "2020": {"label": "2020 COVID-19 Recession",
             "nber_start": "2020-02-01", "nber_end": "2020-04-01",
             "baseline_months": 12, "peak_search_months": 12,
             "recovery_months": 6, "lfpr_months": 24,
             "housing_window": None},
}
# +1 = higher is better, -1 = higher is worse (each feature is z-scored across states)
SCORE_COMPONENTS = {
    "2008": {"ur_shock_2008": -1, "ur_recovery_2008": +1, "lfpr_change_2008": +1},
    "2020": {"ur_shock_2020": -1, "ur_recovery_2020": +1, "lfpr_change_2020": +1},
}
CLUSTER_FEATURES = ["ur_shock_2008", "ur_recovery_2008", "lfpr_change_2008",
                    "ur_shock_2020", "ur_recovery_2020", "lfpr_change_2020"]
CAP = 3.0            # z-scores capped at +/-3 so no single state dominates (e.g. MD)
SEED, N_INIT = 0, 50  # fixed so every viewer sees identical clusters
INCOME_YEARS_AFTER = 4

GREEN, RED, AMBER = "#2e9e5b", "#d64545", "#e8a21a"
DEEP_BLUE, SKY_BLUE = "#1F4E9C", "#8FD0F5"      # two blues that differ strongly in lightness (accessibility)
BLUE = DEEP_BLUE
PALETTE = {3: [GREEN, DEEP_BLUE, RED], 4: [GREEN, SKY_BLUE, DEEP_BLUE, RED]}
QUADS = {"Strong in 2008 and 2020": GREEN, "Strong in 2020 but weak in 2008": DEEP_BLUE,
         "Weak in 2008 and 2020": RED, "Strong in 2008 but weak in 2020": SKY_BLUE}
CAVEAT = ("In clusters-only view, shaded regions outline each cluster's states in these two dimensions. "
          "Clusters were fitted on six features, so the regions overlap here more than they do in the full analysis.")
OHIO_NOTE = ("Clusters group states with similar profiles across six measures and are named by the cluster's average "
             "resilience score. A state's rank (1 to 51) reflects its own overall score, so individual members can rank "
             "higher or lower than their cluster's name suggests.")
NAMES = {3: ["Most resilient", "Middle", "Least resilient"],
         4: ["Most resilient", "Upper-middle", "Lower-middle", "Least resilient"]}
FONT, LABEL_FONT, MARKER, RING = 14, 13, 13, 64   # chart text size, state-label size, dot size, highlight-ring size
MD_COLOR = {3: [":green", ":blue", ":red"], 4: [":green", ":blue", ":blue", ":red"]}

LABELS = {"ur_shock_2008": "2008 unemployment shock", "ur_recovery_2008": "2008 recovery speed",
          "lfpr_change_2008": "2008 participation change", "ur_shock_2020": "2020 unemployment shock",
          "ur_recovery_2020": "2020 recovery speed", "lfpr_change_2020": "2020 participation change"}
ZCOLS = [c for comps in SCORE_COMPONENTS.values() for c in comps]

MAIN_Q = ("Which groups of U.S. states experienced and recovered from the 2008 housing crisis and the "
          "2020 COVID-19 recession in similar ways, and which proved most resilient?")
Q1 = "How do states differ in the size of the initial shock and the speed of recovery in each recession?"
Q2 = ("Are states that were resilient in 2008 also resilient in 2020, or does resilience depend on the "
      "type of shock?")
Q3 = "Which states were most and least resilient overall, and how do they compare to the U.S.?"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading data...")
def read_path(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Cleaning data...")
def clean_long(raw):
    """Parse dates and trim the monthly series to a common end date (documented wrangling step)."""
    long = raw.copy()
    long["date"] = pd.to_datetime(long["date"])
    monthly = ["unemployment_rate", "labor_participation"]
    sm = long[(long.state != "US") & long.indicator.isin(monthly)]
    cutoff = sm.groupby(["indicator", "state"])["date"].max().min()
    long = long[~(long.indicator.isin(monthly) & (long.date > cutoff))].copy()
    return long, cutoff


def wide(long, indicator):
    d = long[long.indicator == indicator]
    return d.pivot(index="date", columns="state", values="value").sort_index()


# ----------------------------------------------------------------------------
# FEATURES AND RESILIENCE SCORE
# ----------------------------------------------------------------------------
def build_features(long):
    ur, lf = wide(long, "unemployment_rate"), wide(long, "labor_participation")
    hpi, inc = wide(long, "house_price_index"), wide(long, "per_capita_income")
    out = pd.DataFrame(index=ur.columns)
    out.index.name = "state"
    for key, cfg in RECESSIONS.items():
        start = pd.Timestamp(cfg["nber_start"])
        b0, b1 = start - pd.DateOffset(months=cfg["baseline_months"]), start - pd.DateOffset(months=1)
        win = ur.loc[start:start + pd.DateOffset(months=cfg["peak_search_months"])]
        peak_dt, peak = win.idxmax(), win.max()
        base = ur.loc[b0:b1].mean()
        shock = peak - base
        lf_base = lf.loc[b0:b1].mean()
        rec, lfc = {}, {}
        for s in ur.columns:
            end = peak_dt[s] + pd.DateOffset(months=cfg["recovery_months"])
            lf_end = peak_dt[s] + pd.DateOffset(months=cfg.get("lfpr_months", cfg["recovery_months"]))
            rec[s] = (peak[s] - ur[s].get(end, np.nan)) / shock[s] if shock[s] > 0 else np.nan
            lfc[s] = lf[s].get(lf_end, np.nan) - lf_base[s]
        out[f"ur_baseline_{key}"], out[f"ur_peak_{key}"] = base, peak
        out[f"ur_peak_date_{key}"], out[f"ur_shock_{key}"] = peak_dt, shock
        out[f"ur_recovery_{key}"] = pd.Series(rec).clip(0, 1)
        out[f"lfpr_change_{key}"] = pd.Series(lfc)
        if cfg.get("housing_window"):
            h = hpi.loc[cfg["housing_window"][0]:cfg["housing_window"][1]]
            out[f"hpi_drop_{key}"] = pd.Series(
                {s: (h[s].loc[h[s].idxmax():].min() / h[s].max() - 1) * 100 for s in h.columns})
        y0 = pd.Timestamp(start.year - 1, 1, 1)
        y1 = pd.Timestamp(start.year - 1 + INCOME_YEARS_AFTER, 1, 1)
        if y0 in inc.index and y1 in inc.index:
            out[f"income_growth_{key}"] = (inc.loc[y1] / inc.loc[y0] - 1) * 100
    return out


def add_resilience(f):
    f = f.copy()
    for key, comps in SCORE_COMPONENTS.items():
        zs = []
        for col, sign in comps.items():
            f["z_" + col] = (f[col] - f[col].mean()) / f[col].std() * sign
            zs.append(f["z_" + col])
        f[f"resilience_{key}"] = pd.concat(zs, axis=1).mean(axis=1)
        f[f"rank_{key}"] = f[f"resilience_{key}"].rank(ascending=False, method="min").astype(int)
    f["resilience_overall"] = f[[f"resilience_{k}" for k in SCORE_COMPONENTS]].mean(axis=1)
    f["rank_overall"] = f["resilience_overall"].rank(ascending=False, method="min").astype(int)
    return f


@st.cache_data(show_spinner="Building features and resilience scores...")
def compute_all(long):
    allf = build_features(long)
    bench = allf.loc["US"]
    states = allf.drop(index="US")
    bench_z = {}
    for comps in SCORE_COMPONENTS.values():
        for col, sign in comps.items():
            bench_z["z_" + col] = (bench[col] - states[col].mean()) / states[col].std() * sign
    return add_resilience(states), bench, pd.Series(bench_z)


# ----------------------------------------------------------------------------
# CLUSTERING AND VALIDATION
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Clustering states...")
def fit_clusters(feats, k, exclude=(), robust=False, n_init=N_INIT):
    """Cluster on the six outcome features. States in `exclude` are not used to fit the
    clusters, but are still assigned to the nearest cluster so they stay on the map."""
    fit_idx = [s for s in feats.index if s not in exclude]
    X = feats[CLUSTER_FEATURES]
    scaler = (RobustScaler() if robust else StandardScaler()).fit(X.loc[fit_idx])
    Z = pd.DataFrame(np.clip(scaler.transform(X), -CAP, CAP), index=feats.index, columns=CLUSTER_FEATURES)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=SEED).fit(Z.loc[fit_idx])
    raw = pd.Series(model.predict(Z), index=feats.index)
    order = feats["resilience_overall"].loc[fit_idx].groupby(raw.loc[fit_idx]).mean() \
        .sort_values(ascending=False).index
    labels = raw.map({old: new for new, old in enumerate(order)})        # 0 = most resilient
    d = model.transform(Z)
    dist = pd.Series(d[np.arange(len(Z)), raw.values], index=feats.index)
    return labels, dist, Z


def match_moves(a, b):
    """Match cluster labels b->a (Hungarian) and list the states whose cluster changed."""
    ct = pd.crosstab(a, b)
    r, c = linear_sum_assignment(-ct.values)
    mapping = {ct.columns[j]: ct.index[i] for i, j in zip(r, c)}
    mapped = pd.Series(b, index=a.index).map(mapping)
    return list(a.index[(mapped != a).to_numpy()])


@st.cache_data(show_spinner="Running with/without tests...")
def scenario_table(feats, k):
    ref, _, _ = fit_clusters(feats, k)
    scen = {"Full (baseline)": ((), False), "Drop MD": (("MD",), False), "Drop DC": (("DC",), False),
            "Drop MD + DC": (("MD", "DC"), False), "Full, robust scaling": ((), True)}
    rows = []
    for name, (drop, robust) in scen.items():
        sub = feats.drop(index=list(drop))
        lab, _, Zs = fit_clusters(sub, k, (), robust)
        best_k = max(range(2, 9), key=lambda kk: silhouette_score(Zs, fit_clusters(sub, kk, (), robust)[0]))
        rows.append({"Scenario": name, "States": len(sub),
                     "ARI vs full": round(adjusted_rand_score(ref.loc[sub.index], lab), 2),
                     "Silhouette": round(silhouette_score(Zs, lab), 2), "Best k": best_k,
                     "States that moved": ", ".join(match_moves(ref.loc[sub.index], lab)) or "-"})
    return pd.DataFrame(rows).set_index("Scenario")


@st.cache_data(show_spinner="Running leave-one-out test (about 10 seconds)...")
def leave_one_out(feats, k):
    ref, _, _ = fit_clusters(feats, k)
    res = {}
    for s in feats.index:
        sub = feats.drop(index=s)
        lab, _, _ = fit_clusters(sub, k, (), False, 20)
        res[s] = adjusted_rand_score(ref.loc[sub.index], lab)
    return pd.Series(res).sort_values()


@st.cache_data(show_spinner=False)
def seed_stability(feats, k):
    ref, _, Z = fit_clusters(feats, k)
    single = [adjusted_rand_score(ref, KMeans(k, n_init=1, random_state=s).fit_predict(Z)) for s in range(50)]
    final = [adjusted_rand_score(ref, KMeans(k, n_init=N_INIT, random_state=s).fit_predict(Z)) for s in range(10)]
    ward = adjusted_rand_score(ref, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z))
    return {"single_mean": np.mean(single), "single_min": np.min(single),
            "final_mean": np.mean(final), "final_min": np.min(final), "ward": ward}


@st.cache_data(show_spinner=False)
def k_diagnostics(feats):
    _, _, Z = fit_clusters(feats, 3)
    rows = []
    for k in range(2, 9):
        m = KMeans(k, n_init=N_INIT, random_state=SEED).fit(Z)
        rows.append({"k": k, "inertia": m.inertia_,
                     "silhouette_kmeans": silhouette_score(Z, m.labels_),
                     "silhouette_ward": silhouette_score(Z, AgglomerativeClustering(k, linkage="ward").fit_predict(Z))})
    return pd.DataFrame(rows).set_index("k")


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
raw = None
for p in DATA_PATHS:
    if os.path.exists(p):
        raw = read_path(p)
        data_src = p
        break
if raw is None:
    st.title("State Economic Resilience Dashboard")
    st.warning("Could not find `data/fred_state_panel_long.csv`. Upload it below "
               "(the CSV saved at the end of the Colab notebook).")
    up = st.file_uploader("Upload fred_state_panel_long.csv", type="csv")
    if up is None:
        st.stop()
    raw, data_src = pd.read_csv(up), "uploaded file"

long, cutoff = clean_long(raw)
feats, bench, bench_z = compute_all(long)
UR = wide(long, "unemployment_rate")
all_states = sorted(feats.index)

# ----------------------------------------------------------------------------
# SIDEBAR (customization)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Customize")
    k = st.radio("Number of clusters (k)", [3, 4], horizontal=True, key="k",
                 help="k=3 is the default (most stable). k=4 splits the middle into two groups.")
    exclude = st.multiselect("Leave states out of the cluster fitting", all_states, default=[], key="excl",
                             format_func=lambda s: f"{s} · {STATE_NAMES[s]}",
                             help="They are not used to build the clusters but are still assigned to the "
                                  "nearest one. Try MD or DC to see why they are kept.")
    robust = st.checkbox("Robust scaling (limits extreme states)", value=False, key="robust",
                         help="Uses median/IQR instead of mean/standard deviation before clustering.")
    hl = st.selectbox("Highlight a state", all_states, index=all_states.index("CA") if "CA" in all_states else 0,
                      format_func=lambda s: f"{s} · {STATE_NAMES[s]}", key="hl")
    detail = st.radio("Chart detail (scatter plots)", ["All states", "Key states", "Clusters only"], key="detail",
                      help="All states: every dot labeled. Key states: only the highlighted state, best/worst, MD, DC and "
                           "any states left out of the fitting are labeled. Clusters only: shaded cluster outlines plus "
                           "those key states.")
    st.caption(CAVEAT)
    st.divider()
    st.caption(f"Data: {data_src}. Monthly series trimmed to {cutoff:%b %Y}.")
    st.caption("Clusters are fitted on 6 outcome features, capped at ±3 SD. Housing and income are used only "
               "to describe clusters.")

labels, dist, Z = fit_clusters(feats, k, tuple(exclude), robust)
fit_idx = [s for s in feats.index if s not in exclude]
cn, cc = NAMES[k], PALETTE[k]
cmap = dict(zip(cn, cc))
df = feats.copy()
df["cluster"], df["cluster_name"], df["dist"] = labels, labels.map(dict(enumerate(cn))), dist
df["name"] = [STATE_NAMES.get(s, s) for s in df.index]
df["state"] = df.index
ranked = df.sort_values("resilience_overall", ascending=False)
best_state, worst_state = ranked.index[0], ranked.index[-1]
sil = silhouette_score(Z.loc[fit_idx], labels.loc[fit_idx])
corr_res = df["resilience_2008"].corr(df["resilience_2020"])


# ----------------------------------------------------------------------------
# CHART HELPERS
# ----------------------------------------------------------------------------
def sign_colors(vals):
    return [GREEN if v >= 0 else RED for v in vals]


def rq_box(label, question, answer=None):
    st.info(f"**{label}:** {question}")
    if answer:
        st.success(answer)


def add_highlight(fig, x, y, color="black"):
    """Ring around a point, sized to also enclose the state label sitting above the dot."""
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=RING, color="rgba(0,0,0,0)", line=dict(width=3, color=color))))


def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


def key_states():
    ks = {hl, best_state, worst_state, "MD", "DC"} | set(exclude)
    return {s for s in ks if s in df.index}


def region_trace(xs, ys, color, name):
    """Shaded outline (convex hull) of a cluster's states; falls back to a line if there are too few points."""
    pts = np.column_stack([xs, ys])
    try:
        hv = ConvexHull(pts).vertices
        hx, hy = np.append(pts[hv, 0], pts[hv[0], 0]), np.append(pts[hv, 1], pts[hv[0], 1])
        return go.Scatter(x=hx, y=hy, mode="lines", fill="toself", fillcolor=hex_rgba(color, 0.15),
                          line=dict(color=color, width=2), name=name, hoverinfo="skip", legendgroup=name)
    except Exception:
        return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3), name=name,
                          hoverinfo="skip", legendgroup=name)


def make_scatter(d, x, y, color_col, color_map, order):
    """Scatter driven by the global 'Chart detail' setting (all states / key states / clusters only)."""
    key = key_states()
    d = d.copy()
    d["label"] = d["state"].where(d["state"].isin(key) | (detail == "All states"), "")
    if detail == "Clusters only":
        d_dots, color_col, color_map, order = d[d["state"].isin(key)], "cluster_name", cmap, cn
    else:
        d_dots = d
    fig = px.scatter(d_dots, x=x, y=y, color=color_col, color_discrete_map=color_map,
                     category_orders={color_col: order}, text="label", hover_name="name")
    fig.update_traces(marker_size=MARKER, textposition="top center", textfont_size=LABEL_FONT)
    if detail == "Clusters only":
        hulls = [region_trace(g[x].values, g[y].values, cmap[n], n)
                 for n in cn for g in [d[d["cluster_name"] == n]] if len(g)]
        for t in fig.data:
            t.showlegend = False
        n_dots = len(fig.data)
        fig.add_traces(hulls)
        fig.data = fig.data[n_dots:] + fig.data[:n_dots]      # draw the shaded regions underneath the dots
    return fig


def big_fonts(fig, legend_below=False):
    fig.update_layout(font=dict(size=FONT), legend=dict(font=dict(size=FONT)), title_font_size=FONT + 4)
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
                          margin=dict(b=90))
    return fig


def map_fig(color_by):
    d = df.reset_index(drop=True)
    hover = {"state": False, "cluster_name": True, "rank_overall": True, "resilience_overall": ":.2f"}
    if color_by == "Cluster":
        fig = px.choropleth(d, locations="state", locationmode="USA-states", color="cluster_name",
                            color_discrete_map=cmap, category_orders={"cluster_name": cn}, scope="usa",
                            hover_name="name", hover_data=hover)
    else:
        fig = px.choropleth(d, locations="state", locationmode="USA-states", color="resilience_overall",
                            color_continuous_scale="RdYlGn", color_continuous_midpoint=0, scope="usa",
                            hover_name="name", hover_data=hover)
    fig.add_trace(go.Choropleth(locations=[hl], z=[1], locationmode="USA-states", showscale=False,
                                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                marker_line_color="black", marker_line_width=3, hoverinfo="skip"))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430, legend_title_text="")
    return fig


def heat_fig():
    g = df.groupby("cluster")[["z_" + c for c in ZCOLS]].mean()
    n = df["cluster"].value_counts().sort_index()
    fig = go.Figure(go.Heatmap(
        z=g.values, x=[LABELS[c] for c in ZCOLS], y=[f"{cn[i]} ({n[i]})" for i in g.index],
        colorscale="RdYlGn", zmid=0, zmin=-1.5, zmax=1.5, text=np.round(g.values, 2),
        texttemplate="%{text}", colorbar=dict(title="vs avg")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430)
    return fig


def ur_fig(states, rec_key, mode, height=430):
    cfg = RECESSIONS[rec_key]
    start, end = pd.Timestamp(cfg["nber_start"]), pd.Timestamp(cfg["nber_end"])
    x0 = start - pd.DateOffset(months=cfg["baseline_months"])
    x1 = start + pd.DateOffset(months=cfg["peak_search_months"] + cfg["recovery_months"])
    base = UR.loc[x0:start - pd.DateOffset(months=1)].mean()
    change = mode.startswith("Change")
    pal = px.colors.qualitative.Safe
    fig = go.Figure()
    fig.add_shape(type="rect", xref="x", yref="paper", x0=f"{start:%Y-%m-%d}", x1=f"{end:%Y-%m-%d}",
                  y0=0, y1=1, fillcolor="rgba(214,69,69,0.13)", line_width=0, layer="below")
    fig.add_annotation(x=f"{start:%Y-%m-%d}", y=1, yref="paper", text="NBER recession", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=RED))
    for i, s in enumerate(states + ["US"]):
        y = UR.loc[x0:x1, s]
        y = y - base[s] if change else y
        is_us = s == "US"
        fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="U.S." if is_us else s,
                                 line=dict(width=3 if is_us else 2.4, dash="dash" if is_us else "solid",
                                           color="black" if is_us else pal[i % len(pal)])))
    if change:
        fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      yaxis_title="Change vs pre-recession baseline (pp)" if change else "Unemployment rate (%)")
    return fig


def comp_fig(state, title=None):
    vals = [df.loc[state, "z_" + c] for c in ZCOLS]
    fig = go.Figure(go.Bar(x=vals, y=[LABELS[c] for c in ZCOLS], orientation="h", marker_color=sign_colors(vals),
                           text=[f"{v:+.2f}" for v in vals], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[-3.6, 3.6], zeroline=True, zerolinecolor="gray")
    fig.update_layout(height=330, margin=dict(l=0, r=10, t=30 if title else 10, b=0), title=title)
    return fig


def metric_label(col_base, rec_key):
    return f"{col_base}_{rec_key}"


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("State Economic Resilience: 2008 Housing Crisis vs. 2020 COVID-19 Recession")
st.caption("Which states were hit similarly, recovered similarly, and proved most resilient? "
           "50 states + DC, unsupervised clustering of FRED labor-market data.")

with st.expander("Research questions and where to find the answers", expanded=False):
    st.markdown(f"**Main question:** {MAIN_Q}  \n→ *Main question tab*\n\n"
                f"**Q1:** {Q1}  \n→ *Q1 tab*\n\n**Q2:** {Q2}  \n→ *Q2 tab*\n\n**Q3:** {Q3}  \n→ *Q3 tab*\n\n"
                "The *Why keep MD & DC?* tab shows the k=3 vs k=4 comparison and the stability tests.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Units clustered", f"{len(df)}", "50 states + DC")
m2.metric("Most resilient", best_state, f"score {df.loc[best_state, 'resilience_overall']:+.2f}")
m3.metric("Least resilient", worst_state, f"score {df.loc[worst_state, 'resilience_overall']:+.2f}")
m4.metric(f"{hl} rank", f"{df.loc[hl, 'rank_overall']} of {len(df)}", df.loc[hl, "cluster_name"], delta_color="off")
m5.metric("2008 vs 2020 resilience (r)", f"{corr_res:.2f}")

tab_main, tab_q1, tab_q2, tab_q3, tab_rob, tab_method = st.tabs(
    ["Main question: state groups", "Q1 · Shock & recovery", "Q2 · Consistency", "Q3 · Best, worst & compare",
     "Why keep MD & DC? (k & robustness)", "Method & data"])

# ----------------------------------------------------------------------------
# TAB: MAIN QUESTION
# ----------------------------------------------------------------------------
with tab_main:
    sizes = df["cluster"].value_counts().sort_index()
    strength = "weak" if sil < 0.25 else ("moderate" if sil < 0.5 else "strong")
    ans = (f"States fall into **{k} groups**: " + ", ".join(f"{cn[i]} ({sizes[i]} states)" for i in sizes.index) +
           f". {hl} is in the **{df.loc[hl, 'cluster_name']}** group. Cluster separation is **{strength}** "
           f"(silhouette {sil:.2f}), so read the groups as overlapping profiles rather than sharply separate types.")
    rq_box("Main question", MAIN_Q, ans)
    c1, c2 = st.columns([3, 2])
    with c1:
        color_by = st.radio("Color the map by", ["Cluster", "Resilience score"], horizontal=True, key="mapcolor")
        st.plotly_chart(map_fig(color_by), key="map")
        st.caption(f"Black outline = highlighted state ({hl}). DC is too small to see on the map; use hover or the lists below.")
    with c2:
        st.markdown("**Cluster fingerprints** (average standardized score; green = better than the average state, "
                    "red = worse)")
        st.plotly_chart(heat_fig(), key="heat")
    st.markdown("**Members**")
    for i in sizes.index:
        mem = ", ".join(sorted(df.index[df.cluster == i]))
        st.markdown(f"{MD_COLOR[k][i]}[**{cn[i]}** ({sizes[i]})]: {mem}")
    st.info(OHIO_NOTE)
    with st.expander("Cluster averages in original units"):
        cols = CLUSTER_FEATURES + ["hpi_drop_2008", "income_growth_2008", "income_growth_2020", "resilience_overall"]
        prof = df.groupby("cluster_name")[cols].mean().reindex(cn)
        prof.insert(0, "states", sizes.values)
        st.dataframe(prof.round(2).rename(columns=LABELS))

# ----------------------------------------------------------------------------
# TAB: Q1
# ----------------------------------------------------------------------------
with tab_q1:
    parts = []
    for key, cfg in RECESSIONS.items():
        sh, rc = df[f"ur_shock_{key}"], df[f"ur_recovery_{key}"]
        worse = int((sh > bench[f"ur_shock_{key}"]).sum())
        parts.append(f"**{cfg['label']}:** median unemployment shock {sh.median():.1f} pp (range {sh.min():.1f} to "
                     f"{sh.max():.1f}); median {rc.median():.0%} of the shock reversed {cfg['recovery_months']} months "
                     f"after each state's peak. U.S. shock was {bench[f'ur_shock_{key}']:.1f} pp, and {worse} of "
                     f"{len(df)} states had a larger shock.")
    rq_box("Q1", Q1, "\n\n".join(parts))

    st.subheader("Shock size vs. recovery speed")
    sc = st.columns(2)
    for col, key in zip(sc, RECESSIONS):
        with col:
            fig = make_scatter(df.reset_index(drop=True), f"ur_shock_{key}", f"ur_recovery_{key}",
                               "cluster_name", cmap, cn)
            fig.update_layout(title=RECESSIONS[key]["label"])
            fig.add_trace(go.Scatter(x=[bench[f"ur_shock_{key}"]], y=[bench[f"ur_recovery_{key}"]],
                                     mode="markers+text", text=["U.S."], textposition="bottom center",
                                     textfont=dict(size=LABEL_FONT + 2), name="U.S. benchmark",
                                     marker=dict(symbol="diamond", size=MARKER + 7, color="black")))
            add_highlight(fig, df.loc[hl, f"ur_shock_{key}"], df.loc[hl, f"ur_recovery_{key}"])
            fig.update_layout(height=640, margin=dict(l=0, r=0, t=40, b=0), legend_title_text="",
                              xaxis_title="Unemployment shock (pp) → worse",
                              yaxis_title=f"Share of shock reversed ({RECESSIONS[key]['recovery_months']} mo.) → better")
            big_fonts(fig, legend_below=True)
            st.plotly_chart(fig, key=f"q1sc{key}")

    if detail == "Clusters only":
        st.caption(CAVEAT)

    st.subheader("How each state compares with the U.S.")
    r1, r2 = st.columns([1, 2])
    rec_pick = r1.radio("Recession", list(RECESSIONS), format_func=lambda x: RECESSIONS[x]["label"], key="q1rec")
    metric_pick = r2.radio("Measure", ["Shock size (pp)", "Recovery speed", "Participation change (pp)"],
                           horizontal=True, key="q1metric")
    base_col, good_sign = {"Shock size (pp)": ("ur_shock", -1), "Recovery speed": ("ur_recovery", 1),
                           "Participation change (pp)": ("lfpr_change", 1)}[metric_pick]
    col = f"{base_col}_{rec_pick}"
    gap = (df[col] - bench[col]) * good_sign          # positive = better than the U.S.
    gap = gap.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=gap.index, y=gap.values, marker_color=sign_colors(gap.values),
                           hovertemplate="%{x}: %{y:+.2f} vs U.S.<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Better (+) or worse (−) than the U.S.")
    st.plotly_chart(fig, key="q1gap")
    st.caption(f"Green = better than the U.S. on this measure, red = worse. U.S. value: {bench[col]:.2f}. "
               f"{hl} is {'better' if gap[hl] >= 0 else 'worse'} than the U.S. by {abs(gap[hl]):.2f}.")

    st.subheader("Recovery curves")
    d1, d2, d3 = st.columns([2, 1, 1])
    default_states = [s for s in dict.fromkeys([best_state, worst_state, hl])]
    curve_states = d1.multiselect("States", all_states, default=default_states, key="q1states",
                                  format_func=lambda s: f"{s} · {STATE_NAMES[s]}")
    curve_rec = d2.radio("Window", list(RECESSIONS), format_func=lambda x: RECESSIONS[x]["label"], key="q1crec")
    curve_mode = d3.radio("Show", ["Unemployment rate", "Change vs baseline"], key="q1cmode")
    st.plotly_chart(ur_fig(curve_states, curve_rec, curve_mode), key="q1curve")
    st.caption("Dashed black line = U.S. Shaded band = NBER recession dates. The window covers the pre-recession "
               "baseline through the end of the recovery horizon.")
    with st.expander("Table: shock and recovery by state (with U.S. benchmark)"):
        cols = [f"{b}_{k_}" for k_ in RECESSIONS for b in ("ur_shock", "ur_recovery", "lfpr_change")]
        tbl = pd.concat([bench[cols].to_frame("US").T, df[cols].sort_index()])
        st.dataframe(tbl.round(2).rename(columns=LABELS))

# ----------------------------------------------------------------------------
# TAB: Q2
# ----------------------------------------------------------------------------
with tab_q2:
    s08, s20 = df.resilience_2008 >= 0, df.resilience_2020 >= 0
    qn_names = list(QUADS)                      # [strong both, strong 2020/weak 2008, weak both, strong 2008/weak 2020]
    quad = pd.Series(qn_names[2], index=df.index)
    quad[s08 & s20], quad[~s08 & s20], quad[s08 & ~s20] = qn_names[0], qn_names[1], qn_names[3]
    qn = quad.value_counts().reindex(qn_names, fill_value=0)
    r_sh = df["ur_shock_2008"].corr(df["ur_shock_2020"])
    strength2 = "weak" if abs(corr_res) < 0.3 else ("moderate" if abs(corr_res) < 0.6 else "strong")
    ans = (f"The correlation between 2008 and 2020 resilience is **{corr_res:.2f}** ({strength2}). "
           f"{qn[qn_names[0]]} states were strong in both recessions, {qn[qn_names[2]]} were weak in both, "
           f"{qn[qn_names[3]]} were strong in 2008 but weak in 2020, and {qn[qn_names[1]]} were strong in 2020 but "
           f"weak in 2008. The size of the initial unemployment shock correlates at {r_sh:.2f} across the two "
           "recessions. Preliminary: definitions are still being refined.")
    rq_box("Q2", Q2, ans)
    d = df.reset_index(drop=True).assign(group=quad.values)
    fig = make_scatter(d, "resilience_2008", "resilience_2020", "group", QUADS, qn_names)
    add_highlight(fig, df.loc[hl, "resilience_2008"], df.loc[hl, "resilience_2020"])
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_dash="dot")
    corners = {qn_names[0]: (0.99, 0.99, "right", "top"), qn_names[1]: (0.01, 0.99, "left", "top"),
               qn_names[2]: (0.01, 0.01, "left", "bottom"), qn_names[3]: (0.99, 0.01, "right", "bottom")}
    for q, (cx, cy, xa, ya) in corners.items():
        fig.add_annotation(xref="paper", yref="paper", x=cx, y=cy, xanchor=xa, yanchor=ya, showarrow=False,
                           text=f"<b>{q}</b><br>{qn[q]} states", align=xa, bgcolor="rgba(255,255,255,0.88)",
                           bordercolor=QUADS[q], borderwidth=2, borderpad=5, font=dict(size=FONT + 1, color="#222"))
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="",
                      xaxis_title="2008 resilience (standardized; higher = better)",
                      yaxis_title="2020 resilience (standardized; higher = better)")
    big_fonts(fig, legend_below=True)
    st.plotly_chart(fig, key="q2sc")
    if detail == "Clusters only":
        st.caption(CAVEAT)
    st.caption("Strong = above the average state (score of 0 or more) in that recession; weak = below it. "
               "Top-right = strong in both; bottom-left = weak in both; the other two corners = resilience depends "
               "on the type of shock.")
    with st.expander("States in each quadrant"):
        for q in qn_names:
            st.markdown(f"**{q}** ({qn[q]}): " + (", ".join(sorted(quad.index[quad == q])) or "none"))
    mv = (df["rank_2008"] - df["rank_2020"]).sort_values()
    a, b = st.columns(2)
    up_, down_ = mv.tail(5)[::-1], mv.head(5)
    a.markdown("**Biggest improvers (rank gained 2008 → 2020)**")
    a.dataframe(pd.DataFrame({"2008 rank": df.loc[up_.index, "rank_2008"], "2020 rank": df.loc[up_.index, "rank_2020"],
                              "places gained": up_.astype(int)}))
    b.markdown("**Biggest fallers (rank lost 2008 → 2020)**")
    b.dataframe(pd.DataFrame({"2008 rank": df.loc[down_.index, "rank_2008"], "2020 rank": df.loc[down_.index, "rank_2020"],
                              "places lost": (-down_).astype(int)}))
    st.markdown(f"**{hl}:** 2008 rank {df.loc[hl, 'rank_2008']}, 2020 rank {df.loc[hl, 'rank_2020']} (of {len(df)}).")

# ----------------------------------------------------------------------------
# TAB: Q3
# ----------------------------------------------------------------------------
with tab_q3:
    top5, bot5 = list(ranked.index[:5]), list(ranked.index[-5:][::-1])
    ans = (f"**Most resilient:** {', '.join(top5)}. **Least resilient:** {', '.join(bot5)}. "
           f"{hl} ranks {df.loc[hl, 'rank_overall']} of {len(df)}. The score averages standardized shock, recovery "
           "and participation change across both recessions; use the comparison below to see how any state "
           "stacks up against the U.S. benchmark.")
    rq_box("Q3", Q3, ans)
    rk = ranked.iloc[::-1].reset_index(drop=True)
    fig = go.Figure(go.Bar(x=rk["resilience_overall"], y=rk["state"], orientation="h",
                           marker_color=[cmap[n] for n in rk["cluster_name"]], customdata=rk[["name", "cluster_name"]],
                           hovertemplate="%{customdata[0]} (%{customdata[1]}): %{x:+.2f}<extra></extra>"))
    fig.update_layout(height=1050, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Overall resilience score (higher = better)",
                      yaxis=dict(tickfont=dict(size=10)))
    with st.expander("Full ranking (all states, colored by cluster)", expanded=False):
        st.plotly_chart(fig, key="q3rank")

    st.subheader("Compare any two states")
    st.caption("Defaults to the most and least resilient states. Switch to any pair, for example your home state vs. the best.")
    p1, p2 = st.columns(2)
    sa = p1.selectbox("State A", all_states, index=all_states.index(best_state), key="cmpA",
                      format_func=lambda s: f"{s} · {STATE_NAMES[s]}")
    sb = p2.selectbox("State B", all_states, index=all_states.index(worst_state), key="cmpB",
                      format_func=lambda s: f"{s} · {STATE_NAMES[s]}")
    ca, cb = st.columns(2)
    for col_, s in ((ca, sa), (cb, sb)):
        with col_:
            st.markdown(f"### {STATE_NAMES[s]}")
            x1_, x2_, x3_ = st.columns(3)
            x1_.metric("Overall rank", f"{df.loc[s, 'rank_overall']} of {len(df)}")
            x2_.metric("Score", f"{df.loc[s, 'resilience_overall']:+.2f}")
            x3_.metric("Cluster", df.loc[s, "cluster_name"])
            st.plotly_chart(comp_fig(s), key=f"cmp_{s}_{'a' if s == sa and col_ is ca else 'b'}")
    st.caption("Bars are standardized scores (shock flipped so up is always better): green = better than the average state, red = worse.")

    diff = pd.Series({c: df.loc[sa, "z_" + c] - df.loc[sb, "z_" + c] for c in ZCOLS})
    fig = go.Figure(go.Bar(x=diff.values, y=[LABELS[c] for c in diff.index], orientation="h",
                           marker_color=sign_colors(diff.values), text=[f"{v:+.2f}" for v in diff.values],
                           textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=0, r=10, t=30, b=0),
                      title=f"Where {sa} does better (green) or worse (red) than {sb}")
    st.plotly_chart(fig, key="cmp_diff")

    raw_cols = CLUSTER_FEATURES + ["hpi_drop_2008", "income_growth_2008", "income_growth_2020"]
    cmp_tbl = pd.DataFrame({sa: df.loc[sa, raw_cols], sb: df.loc[sb, raw_cols], "U.S. benchmark": bench[raw_cols]})
    cmp_tbl.index = [LABELS.get(c, c.replace("_", " ")) for c in raw_cols]
    st.dataframe(cmp_tbl.astype(float).round(2))
    cr = st.radio("Recovery curves for this pair", list(RECESSIONS), format_func=lambda x: RECESSIONS[x]["label"],
                  horizontal=True, key="cmprec")
    st.plotly_chart(ur_fig([sa, sb], cr, "Change vs baseline"), key="cmpcurve")

# ----------------------------------------------------------------------------
# TAB: ROBUSTNESS
# ----------------------------------------------------------------------------
with tab_rob:
    st.markdown("**Why Maryland and DC stay in the analysis.** Use the sidebar to switch k, leave states out of the "
                "fitting, or use robust scaling. The tests below always compare against the full 51-unit clustering.")
    if exclude or robust:
        ref_full, _, _ = fit_clusters(feats, k)
        moved = match_moves(ref_full, labels)
        st.warning(f"Current sidebar settings (k={k}, left out: {', '.join(exclude) or 'none'}, "
                   f"robust scaling: {'on' if robust else 'off'}) match the full clustering with ARI "
                   f"{adjusted_rand_score(ref_full, labels):.2f}. States whose cluster changed: {', '.join(moved) or 'none'}.")
    run_tests = st.toggle("Run the stability tests (about 30 to 90 seconds the first time, instant afterwards)",
                          value=False, key="run_tests",
                          help="These refit the clusters hundreds of times, so they only run when you switch this on.")

    if run_tests:
        st.subheader(f"1. With / without MD and DC (k = {k})")
        st.dataframe(scenario_table(feats, k))
        st.caption("ARI = agreement with the full clustering (1.00 = identical). Dropping any state moves things a little, "
                   "so compare with the leave-one-out result below. 'Best k' by silhouette is high in every scenario, "
                   "another sign that cluster structure is weak.")
    else:
        st.subheader(f"1. With / without MD and DC (k = {k})")
        st.caption("Switch on \"Run the stability tests\" above to compute this section.")

    if run_tests:
        st.subheader("2. Is MD or DC unusually influential? (leave-one-out)")
        loo = leave_one_out(feats, k)
        top = loo.head(15)
        fig = go.Figure(go.Bar(x=top.index, y=top.values,
                               marker_color=[AMBER if s in ("MD", "DC") else BLUE for s in top.index]))
        fig.add_hline(y=loo.median(), line_dash="dot", line_color="gray",
                      annotation_text=f"median state {loo.median():.2f}", annotation_font_size=FONT)
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="ARI when state is dropped (lower = more influential)")
        big_fonts(fig)
        st.plotly_chart(fig, key="loo")
        rank = loo.rank(method="min")
        st.markdown(f"**MD:** ARI {loo['MD']:.2f}, rank {int(rank['MD'])} of {len(loo)} (1 = most influential). "
                    f"**DC:** ARI {loo['DC']:.2f}, rank {int(rank['DC'])} of {len(loo)}. "
                    f"Median state: {loo.median():.2f}. Amber bars = MD / DC (only shown if among the 15 most influential).")
    else:
        st.subheader("2. Is MD or DC unusually influential? (leave-one-out)")
        st.caption("Switch on \"Run the stability tests\" above to compute this section.")

    st.subheader("3. How well does each state fit its cluster?")
    far = df["dist"].sort_values(ascending=False).head(12)
    fig = go.Figure(go.Bar(x=far.index, y=far.values, marker_color=[AMBER if s in ("MD", "DC") else BLUE for s in far.index]))
    fig.add_hline(y=df["dist"].median(), line_dash="dot", line_color="gray",
                  annotation_text=f"median state {df['dist'].median():.2f}", annotation_font_size=FONT)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Distance to own cluster center")
    big_fonts(fig)
    st.plotly_chart(fig, key="dist")
    st.caption(f"MD: {df.loc['MD', 'dist']:.2f}, DC: {df.loc['DC', 'dist']:.2f}. A large distance means a state is assigned "
               "to a cluster but is atypical in it. MD's 2020 participation drop (a one-month, non-reversing step) is "
               "the main reason.")

    if run_tests:
        st.subheader("4. k = 3 vs. k = 4")
        l3, _, _ = fit_clusters(feats, 3)
        l4, _, _ = fit_clusters(feats, 4)
        s3, s4 = seed_stability(feats, 3), seed_stability(feats, 4)
        n3, n4 = sorted(l3.value_counts().tolist(), reverse=True), sorted(l4.value_counts().tolist(), reverse=True)
        sc3, sc4 = scenario_table(feats, 3), scenario_table(feats, 4)
        cmp_k = pd.DataFrame({
            "Cluster sizes": [str(n3), str(n4)],
            "K-means vs Ward (ARI)": [round(s3["ward"], 2), round(s4["ward"], 2)],
            "Drop MD (ARI)": [sc3.loc["Drop MD", "ARI vs full"], sc4.loc["Drop MD", "ARI vs full"]],
            "Drop MD: states moved": [sc3.loc["Drop MD", "States that moved"], sc4.loc["Drop MD", "States that moved"]],
            "Robust scaling (ARI)": [sc3.loc["Full, robust scaling", "ARI vs full"], sc4.loc["Full, robust scaling", "ARI vs full"]],
            "Silhouette": [sc3.loc["Full (baseline)", "Silhouette"], sc4.loc["Full (baseline)", "Silhouette"]],
        }, index=["k = 3", "k = 4"])
        st.dataframe(cmp_k)
        mem = pd.DataFrame({"k=3": l3.map(dict(enumerate(NAMES[3]))), "k=4": l4.map(dict(enumerate(NAMES[4])))})
        st.markdown("**How states move between k=3 and k=4** (rows = k=3 cluster, columns = k=4 cluster)")
        st.dataframe(pd.crosstab(mem["k=3"], mem["k=4"]).reindex(index=NAMES[3], columns=NAMES[4], fill_value=0))
        with st.expander("State-by-state cluster membership for k=3 and k=4"):
            st.dataframe(mem.sort_index())
    else:
        st.subheader("4. k = 3 vs. k = 4")
        st.caption("Switch on \"Run the stability tests\" above to compute this section.")

    if run_tests:
        st.subheader(f"5. Stability of the current clustering (k = {k})")
        ss = seed_stability(feats, k)
        s1, s2, s3_ = st.columns(3)
        s1.metric("Final model across 10 seeds (mean ARI)", f"{ss['final_mean']:.2f}", f"min {ss['final_min']:.2f}", delta_color="off")
        s2.metric("Single random starts (mean ARI)", f"{ss['single_mean']:.2f}", f"min {ss['single_min']:.2f}", delta_color="off")
        s3_.metric("Agreement with Ward clustering (ARI)", f"{ss['ward']:.2f}")
    else:
        st.subheader(f"5. Stability of the current clustering (k = {k})")
        st.caption("Switch on \"Run the stability tests\" above to compute this section.")

    if run_tests:
        st.subheader("6. Choosing k")
        kd = k_diagnostics(feats)
        f1, f2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=kd.index, y=kd.inertia, mode="lines+markers"))
        fig.update_layout(height=360, title="Elbow (inertia vs k): no clear bend", margin=dict(l=0, r=0, t=50, b=0), xaxis_title="k")
        big_fonts(fig)
        f1.plotly_chart(fig, key="elbow")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kd.index, y=kd.silhouette_kmeans, mode="lines+markers", name="k-means"))
        fig.add_trace(go.Scatter(x=kd.index, y=kd.silhouette_ward, mode="lines+markers", name="Ward"))
        fig.update_layout(height=360, title="Silhouette vs k (all low)", margin=dict(l=0, r=0, t=50, b=0), xaxis_title="k")
        big_fonts(fig)
        f2.plotly_chart(fig, key="sil")
    else:
        st.subheader("6. Choosing k")
        st.caption("Switch on \"Run the stability tests\" above to compute this section.")

    st.subheader("7. Cluster map in two dimensions (PCA)")
    pca = PCA(2).fit(Z)
    P = pd.DataFrame(pca.transform(Z), index=Z.index, columns=["PC1", "PC2"]).join(df[["cluster_name", "name"]])
    fig = make_scatter(P.reset_index(), "PC1", "PC2", "cluster_name", cmap, cn)
    for s_ in ("MD", "DC"):
        add_highlight(fig, P.loc[s_, "PC1"], P.loc[s_, "PC2"], AMBER)
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="",
                      xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
                      yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    big_fonts(fig, legend_below=True)
    st.plotly_chart(fig, key="pca")
    if detail == "Clusters only":
        st.caption(CAVEAT)
    st.caption("Amber rings = MD and DC. The two axes show only about half of the variance, so a state can look central here "
               "and still be far from its cluster center in the full six dimensions.")

# ----------------------------------------------------------------------------
# TAB: METHOD
# ----------------------------------------------------------------------------
with tab_method:
    st.markdown("### Method summary")
    st.markdown(
        "1. **Data:** FRED state series (unemployment rate, labor force participation, house price index, per "
        "capita income) for 50 states + DC, plus U.S. benchmarks. The U.S. is a reference, not a clustered unit.\n"
        "2. **Windows:** each recession is a config entry (below); the full data is never cut.\n"
        "3. **Features (one row per state):** unemployment shock = peak minus 12-month pre-recession baseline; "
        "recovery = share of the shock reversed a set number of months after the state's own peak (clipped 0 to 1); "
        "participation change = level after the horizon minus baseline.\n"
        "4. **Resilience score:** z-score each component across states, flip shock so higher is better, average within "
        "each recession, then average the two recessions. Equal weights.\n"
        "5. **Clustering:** k-means (fixed seed, 50 starts) on the six outcome features, standardized and capped at "
        "±3 SD. Housing and income describe the clusters but do not build them. Clusters are numbered by average "
        "resilience (green = most resilient).")
    win = pd.DataFrame(RECESSIONS).T.drop(columns=["housing_window"]).rename(columns={
        "label": "Recession", "nber_start": "NBER start", "nber_end": "NBER end", "baseline_months": "Baseline (mo)",
        "peak_search_months": "Peak search (mo)", "recovery_months": "Recovery check (mo)", "lfpr_months": "Participation check (mo)"})
    st.markdown("### Recession windows")
    st.dataframe(win.set_index("Recession"))
    st.markdown("### Data coverage")
    cov = long.groupby("indicator").agg(series=("state", "nunique"), rows=("value", "count"),
                                        first=("date", "min"), last=("date", "max"))
    st.dataframe(cov)
    st.markdown(f"### Wrangling notes\n"
                f"- Monthly series trimmed to a common end date ({cutoff:%B %Y}). No internal gaps and no missing "
                "states or indicators were found; no imputation was needed.\n"
                "- Mixed frequencies handled by summarizing each recession into one number per state.\n"
                "- Maryland's 2020 participation drop is a one-month, non-reversing step; kept, capped at ±3 SD, and tested with and "
                "without it.\n"
                "- Outlier handling is tested in the *Why keep MD & DC?* tab.")
    st.markdown("### Known limitations")
    st.markdown("- Weak cluster structure (silhouette about 0.2): groups overlap.\n"
                "- The 5-state extreme cluster at k=3 depends on each of its members.\n"
                "- Recovery horizons (24 months for 2008, 6 for 2020) are judgment calls.\n"
                "- Recovery is clipped at 1; participation change partly reflects aging, not only recession effects.\n"
                "- Income growth is nominal and used for description only.")
    st.markdown("### Glossary")
    st.markdown("- **ARI (adjusted Rand index):** agreement between two clusterings; 1 = identical, about 0 = unrelated.\n"
                "- **Silhouette:** how well separated clusters are; below about 0.25 is generally weak.\n"
                "- **pp:** percentage points.")
