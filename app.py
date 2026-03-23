"""
London Airbnb Price Predictor - Dash Application
Inference app that loads a trained notebook model artifact.
"""

import json
import os
from pathlib import Path
from typing import Dict

import dash
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

DATA_PATH = Path("listings.csv.gz")
MODEL_PATH = Path("artifacts/rf_price_model.joblib")
META_PATH = Path("artifacts/model_metadata.json")


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH.resolve()}. Place listings.csv.gz in the app folder."
    )

df = pd.read_csv(DATA_PATH, compression="gzip")


def parse_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


# Minimal frame for visualizations.
dashboard_df = df.copy()
dashboard_df["price"] = parse_price(dashboard_df["price"])
dashboard_df = dashboard_df.dropna(
    subset=["price", "neighbourhood_cleansed", "room_type"]
).copy()


if not MODEL_PATH.exists() or not META_PATH.exists():
    # 如果在Render上则创建公告
    if os.environ.get("RENDER"):
        raise FileNotFoundError(
            "Model artifacts are missing on Render. "
            "Please upload the model files manually or run the notebook export before deployment."
        )
    raise FileNotFoundError(
        "Model artifacts are missing. Run the model export cell in the notebook first. "
        f"Expected files: {MODEL_PATH.resolve()} and {META_PATH.resolve()}"
    )

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]
default_values = metadata["default_values"]
residual_q05 = float(metadata.get("residual_q05", -50.0))
residual_q95 = float(metadata.get("residual_q95", 50.0))


neighbourhoods_list = sorted(dashboard_df["neighbourhood_cleansed"].dropna().unique())
room_types_list = sorted(dashboard_df["room_type"].dropna().unique())

neighbourhood_stats = dashboard_df.groupby("neighbourhood_cleansed").agg(
    price_count=("price", "count"),
    avg_price=("price", "mean"),
    median_price=("price", "median"),
    avg_review=("review_scores_rating", "mean"),
    avg_bedrooms=("bedrooms", "mean"),
).round(2)


app = dash.Dash(__name__)
server = app.server
app.title = "London Airbnb Price Predictor"

PREDICTION_BOX_BASE_STYLE = {
    "fontSize": "32px",
    "fontWeight": "bold",
    "color": "#ffffff",
    "padding": "20px",
    "borderRadius": "5px",
    "textAlign": "center",
    "marginBottom": "20px",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def score_to_level(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def score_to_color(score: float) -> str:
    # 0 -> deep red, 100 -> deep green.
    t = clamp(score, 0.0, 100.0) / 100.0
    red = int(round(139 * (1 - t) + 0 * t))
    green = int(round(0 * (1 - t) + 100 * t))
    blue = int(round(0 * (1 - t) + 0 * t))
    return f"rgb({red}, {green}, {blue})"


def compute_support_score(scoped_count: int, combo_count: int) -> float:
    exact_coverage = 100.0 * clamp(scoped_count / 20.0, 0.0, 1.0)
    combo_coverage = 100.0 * clamp(combo_count / 50.0, 0.0, 1.0)
    if scoped_count == 0:
        return 0.35 * combo_coverage
    return 0.75 * exact_coverage + 0.25 * combo_coverage


def compute_interval_score(prediction: float, lower_bound: float, upper_bound: float, reference_price: float) -> float:
    width = max(upper_bound - lower_bound, 1.0)
    ref = max(reference_price, prediction, 50.0)
    relative_width = width / ref
    # Relative width <= 0.5 => strong confidence; >= 3.0 => weak confidence.
    normalized = clamp((relative_width - 0.5) / 2.5, 0.0, 1.0)
    return 100.0 * (1.0 - normalized)


def compute_distance_score(input_row: Dict, numeric_stats: Dict[str, Dict[str, float]]) -> float:
    if not numeric_stats:
        return 50.0

    penalties = []
    for col, stats in numeric_stats.items():
        raw_val = input_row.get(col)
        val = pd.to_numeric(pd.Series([raw_val]), errors="coerce").iloc[0]
        if pd.isna(val):
            continue

        std = stats["std"]
        if std <= 1e-9:
            continue

        z = abs(float(val) - stats["median"]) / std
        penalties.append(clamp(z / 3.0, 0.0, 1.0))

    if not penalties:
        return 50.0

    mean_penalty = float(sum(penalties) / len(penalties))
    return 100.0 * (1.0 - mean_penalty)


def compute_plausibility_score(bedrooms: float, accommodates: float) -> float:
    bedrooms = max(float(bedrooms), 1.0)
    accommodates = max(float(accommodates), 1.0)
    ratio = accommodates / bedrooms

    if ratio < 1.0:
        return 0.0
    if ratio < 1.3:
        return 35.0
    if ratio <= 4.0:
        return 100.0
    if ratio <= 6.0:
        return 70.0
    return 40.0


def compute_joint_rarity_score(combo_df: pd.DataFrame, bedrooms: float, accommodates: float) -> float:
    if combo_df.empty:
        return 0.0

    b = float(bedrooms)
    a = float(accommodates)

    b_series = pd.to_numeric(combo_df.get("bedrooms"), errors="coerce")
    a_series = pd.to_numeric(combo_df.get("accommodates"), errors="coerce")
    valid = combo_df.loc[b_series.notna() & a_series.notna()].copy()
    if valid.empty:
        return 20.0

    b_valid = pd.to_numeric(valid["bedrooms"], errors="coerce")
    a_valid = pd.to_numeric(valid["accommodates"], errors="coerce")

    exact_count = int(((b_valid == b) & (a_valid == a)).sum())
    near_count = int(((b_valid.between(b - 1, b + 1)) & (a_valid.between(a - 2, a + 2))).sum())

    exact_score = 100.0 * clamp(exact_count / 10.0, 0.0, 1.0)
    near_score = 100.0 * clamp(near_count / 20.0, 0.0, 1.0)
    return 0.7 * exact_score + 0.3 * near_score


def build_empty_figure(title: str, message: str) -> go.Figure:
    """Create a stable placeholder figure for empty-filter results."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#666"},
            }
        ],
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def build_numeric_feature_stats(frame: pd.DataFrame, columns: list[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in frame.columns:
            continue

        series = frame[col]
        if col == "host_is_superhost":
            series = series.map({"t": 1, "f": 0, True: 1, False: 0})

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) < 30:
            continue

        std = float(numeric_series.std())
        if pd.isna(std) or std <= 1e-9:
            continue

        stats[col] = {
            "median": float(numeric_series.median()),
            "std": std,
        }

    return stats


numeric_feature_stats = build_numeric_feature_stats(dashboard_df, feature_columns)


@app.callback(
    [
        Output("accommodates-slider", "min"),
        Output("accommodates-slider", "value"),
        Output("accommodates-slider", "marks"),
    ],
    [
        Input("bedrooms-slider", "value"),
    ],
    [
        State("accommodates-slider", "value"),
    ],
)
def sync_accommodates_with_bedrooms(bedrooms, current_accommodates):
    min_guests = int(max(1, bedrooms or 1))
    max_guests = 16
    current = int(current_accommodates) if current_accommodates is not None else min_guests
    adjusted_value = int(clamp(current, min_guests, max_guests))

    base_marks = {i: str(i) for i in range(1, 17, 2)}
    base_marks[min_guests] = str(min_guests)
    base_marks[adjusted_value] = str(adjusted_value)
    marks = {k: base_marks[k] for k in sorted(base_marks)}

    return min_guests, adjusted_value, marks

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("London Airbnb Nightly Price Predictor", style={"marginBottom": "10px"}),
                html.P(
                    "This dashboard predicts nightly prices using the trained Random Forest model exported from the notebook.",
                    style={"color": "#444", "fontSize": "14px", "marginBottom": "8px"},
                ),
                html.P(
                    "How to use: choose a neighbourhood and room type, then adjust superhost, bedrooms, and max guests. "
                    "The prediction updates instantly, together with a matched-listings count and supporting charts.",
                    style={"color": "#444", "fontSize": "14px", "marginBottom": "8px"},
                ),
                html.P(
                    "How to read confidence: the Composite Score ranges from 0 to 100 and combines data support, "
                    "distribution distance, interval tightness, plausibility, and rarity. Higher is more reliable.",
                    style={"color": "#444", "fontSize": "14px", "marginBottom": "8px"},
                ),
                html.P(
                    "Color guide: green means higher confidence, amber means medium confidence, and red means lower confidence. "
                    "Low confidence does not mean the prediction is wrong; it means uncertainty is higher.",
                    style={"color": "#444", "fontSize": "14px", "marginBottom": "0"},
                ),
            ],
            style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderBottom": "2px solid #007bff"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Filters", style={"marginTop": "0"}),
                        html.Label("Neighbourhood:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="neighbourhood-dropdown",
                            options=[{"label": n, "value": n} for n in neighbourhoods_list],
                            value=neighbourhoods_list[0] if neighbourhoods_list else None,
                            style={"marginBottom": "20px"},
                        ),
                        html.Label("Room Type:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="room-type-dropdown",
                            options=[{"label": rt, "value": rt} for rt in room_types_list],
                            value=room_types_list[0] if room_types_list else None,
                            style={"marginBottom": "20px"},
                        ),
                        html.Label("Host Is Superhost:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="superhost-dropdown",
                            options=[
                                {"label": "No", "value": 0},
                                {"label": "Yes", "value": 1},
                            ],
                            value=0,
                            clearable=False,
                            style={"marginBottom": "20px"},
                        ),
                        html.Label("Bedrooms:", style={"fontWeight": "bold"}),
                        html.Div(
                            [
                                dcc.Slider(
                                    id="bedrooms-slider",
                                    min=1,
                                    max=6,
                                    step=1,
                                    value=1,
                                    marks={i: str(i) for i in range(1, 7)},
                                )
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Constraint applied: Max Guests must be greater than or equal to Bedrooms.",
                            style={"fontSize": "12px", "color": "#555", "marginTop": "-10px", "marginBottom": "12px"},
                        ),
                        html.Label("Max Guests:", style={"fontWeight": "bold"}),
                        html.Div(
                            [
                                dcc.Slider(
                                    id="accommodates-slider",
                                    min=1,
                                    max=16,
                                    step=1,
                                    value=4,
                                    marks={i: str(i) for i in range(1, 17, 2)},
                                )
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "This rule keeps inputs realistic and helps avoid impossible combinations that can distort predictions.",
                            style={"fontSize": "12px", "color": "#555", "marginTop": "-10px", "marginBottom": "0"},
                        ),
                    ],
                    style={
                        "width": "28%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "20px",
                        "backgroundColor": "#f9f9f9",
                        "borderRight": "1px solid #ddd",
                        "maxHeight": "100vh",
                        "overflowY": "auto",
                        "boxSizing": "border-box",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Price Prediction", id="prediction-title"),
                                html.Div(
                                    id="prediction-output",
                                    style={
                                        **PREDICTION_BOX_BASE_STYLE,
                                        "backgroundColor": "rgb(46, 125, 50)",
                                    },
                                ),
                                html.P(id="prediction-note", style={"fontSize": "12px", "color": "#666"}),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [
                                html.H4("Neighbourhood Summary"),
                                html.Table(
                                    id="neighbourhood-stats-table",
                                    style={
                                        "width": "100%",
                                        "borderCollapse": "collapse",
                                        "marginBottom": "20px",
                                        "fontSize": "13px",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [
                                html.H4("Price Distribution in Selected Scope"),
                                dcc.Graph(id="price-dist-chart"),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [
                                html.H4("Average Price by Room Type (Selected Neighbourhood)"),
                                dcc.Graph(id="room-type-chart"),
                            ]
                        ),
                    ],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "20px",
                        "maxHeight": "100vh",
                        "overflowY": "auto",
                        "boxSizing": "border-box",
                    },
                ),
            ],
            style={"display": "flex"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "margin": "0", "padding": "0"},
)


@app.callback(
    [
        Output("prediction-title", "children"),
        Output("prediction-output", "children"),
        Output("prediction-output", "style"),
        Output("prediction-note", "children"),
        Output("neighbourhood-stats-table", "children"),
        Output("price-dist-chart", "figure"),
        Output("room-type-chart", "figure"),
    ],
    [
        Input("neighbourhood-dropdown", "value"),
        Input("room-type-dropdown", "value"),
        Input("superhost-dropdown", "value"),
        Input("bedrooms-slider", "value"),
        Input("accommodates-slider", "value"),
    ],
)
def update_dashboard(neighbourhood, room_type, superhost_value, bedrooms, accommodates):
    bedrooms = float(bedrooms)
    accommodates = float(accommodates)
    adjusted_accommodates = max(accommodates, bedrooms)

    combo_df = dashboard_df[
        (dashboard_df["neighbourhood_cleansed"] == neighbourhood)
        & (dashboard_df["room_type"] == room_type)
    ].copy()
    scoped_df = combo_df.copy()

    # Build model input from saved defaults + user overrides.
    input_row = {col: default_values.get(col, 0) for col in feature_columns}
    if "neighbourhood_cleansed" in input_row:
        input_row["neighbourhood_cleansed"] = neighbourhood
    if "room_type" in input_row:
        input_row["room_type"] = room_type
    if "bedrooms" in input_row:
        input_row["bedrooms"] = float(bedrooms)
    if "accommodates" in input_row:
        input_row["accommodates"] = float(adjusted_accommodates)
    if "host_is_superhost" in input_row:
        input_row["host_is_superhost"] = float(superhost_value)
    if "is_studio" in input_row:
        input_row["is_studio"] = 1.0 if (room_type == "Entire home/apt" and float(bedrooms) == 0) else 0.0

    selected_superhost_df = scoped_df.copy()
    if "host_is_superhost" in selected_superhost_df.columns:
        superhost_numeric = selected_superhost_df["host_is_superhost"].map(
            {"t": 1, "f": 0, True: 1, False: 0, 1: 1, 0: 0, "1": 1, "0": 0}
        )
        valid_superhost = superhost_numeric.notna()
        if valid_superhost.any():
            scoped_df = selected_superhost_df.loc[
                valid_superhost & (superhost_numeric == float(superhost_value))
            ].copy()

    input_df = pd.DataFrame([input_row], columns=feature_columns)
    prediction = float(model.predict(input_df)[0])

    lower_bound = max(prediction + residual_q05, 20)
    upper_bound = max(prediction + residual_q95, lower_bound)

    pred_text = f"£{max(prediction, 20):.2f}/night"
    superhost_label = "Yes" if int(superhost_value) == 1 else "No"

    ref_price = (
        float(neighbourhood_stats.loc[neighbourhood]["median_price"])
        if neighbourhood in neighbourhood_stats.index
        else float(dashboard_df["price"].median())
    )

    support_score = compute_support_score(len(scoped_df), len(combo_df))
    distance_score = compute_distance_score(input_row, numeric_feature_stats)
    interval_score = compute_interval_score(prediction, lower_bound, upper_bound, ref_price)
    plausibility_score = compute_plausibility_score(bedrooms, adjusted_accommodates)
    rarity_score = compute_joint_rarity_score(combo_df, bedrooms, adjusted_accommodates)

    confidence_score = (
        0.30 * support_score
        + 0.20 * distance_score
        + 0.20 * interval_score
        + 0.15 * plausibility_score
        + 0.15 * rarity_score
    )

    # Hard downgrade rules for logically inconsistent or very sparse combinations.
    if accommodates < bedrooms:
        confidence_score = min(confidence_score, 25.0)
    elif len(scoped_df) <= 1:
        confidence_score = min(confidence_score, 35.0)
    elif len(scoped_df) <= 3:
        confidence_score = min(confidence_score, 50.0)

    if rarity_score < 20.0:
        confidence_score = min(confidence_score, 45.0)

    confidence_score = round(clamp(confidence_score, 0.0, 100.0), 1)
    confidence_level = score_to_level(confidence_score)

    prediction_title = (
        "Price Prediction | "
        f"Current prediction confidence composite score: {confidence_score:.1f}, "
        f"confidence level: {confidence_level}."
    )
    prediction_style = {
        **PREDICTION_BOX_BASE_STYLE,
        "backgroundColor": score_to_color(confidence_score),
    }

    risk_note = (
        "Composite confidence score = 30% data support + 20% distribution distance + "
        "20% interval tightness + 15% feature plausibility + 15% joint rarity. "
        f"Current components: support={support_score:.1f}, distance={distance_score:.1f}, "
        f"interval={interval_score:.1f}, plausibility={plausibility_score:.1f}, rarity={rarity_score:.1f}."
    )

    if accommodates < bedrooms:
        risk_note += " Input adjusted for prediction: accommodates was raised to match bedrooms."

    pred_note = (
        f"Predicted range: £{lower_bound:.2f} — £{upper_bound:.2f}. "
        f"Matched listings: {len(scoped_df)}. "
        f"Scope: neighbourhood + room type (+ superhost filter when available: {superhost_label}). "
        f"{risk_note}"
    )

    if neighbourhood in neighbourhood_stats.index:
        stats = neighbourhood_stats.loc[neighbourhood]
        stats_rows = [
            html.Tr(
                [
                    html.Td("Selection:", style={"fontWeight": "bold", "padding": "8px"}),
                    html.Td(f"{neighbourhood} | {room_type} | Superhost={superhost_label}", style={"padding": "8px"}),
                ]
            ),
            html.Tr(
                [
                    html.Td("Matched Listings:", style={"fontWeight": "bold", "padding": "8px"}),
                    html.Td(f"{len(scoped_df)}", style={"padding": "8px"}),
                ]
            ),
            html.Tr(
                [
                    html.Td("Neighbourhood Listings:", style={"fontWeight": "bold", "padding": "8px"}),
                    html.Td(f"{int(stats['price_count'])}", style={"padding": "8px"}),
                ]
            ),
            html.Tr(
                [
                    html.Td("Neighbourhood Avg Price:", style={"fontWeight": "bold", "padding": "8px"}),
                    html.Td(f"£{stats['avg_price']:.2f}", style={"padding": "8px"}),
                ]
            ),
            html.Tr(
                [
                    html.Td("Neighbourhood Median Price:", style={"fontWeight": "bold", "padding": "8px"}),
                    html.Td(f"£{stats['median_price']:.2f}", style={"padding": "8px"}),
                ]
            ),
        ]
        stats_table = html.Tbody(stats_rows)
    else:
        stats_table = html.Tbody([html.Tr([html.Td("No stats available")])])

    if len(scoped_df) > 0:
        dist_df = scoped_df
        dist_title = f"Price Distribution — {neighbourhood} ({room_type}, Superhost={superhost_label})"
    elif len(combo_df) > 0:
        dist_df = combo_df
        dist_title = f"Price Distribution — {neighbourhood} ({room_type}, all superhost statuses)"
    else:
        dist_df = pd.DataFrame()
        dist_title = "Price Distribution in Selected Scope"

    if len(dist_df) > 0:
        price_dist = px.histogram(
            dist_df,
            x="price",
            nbins=30,
            title=dist_title,
            labels={"price": "Nightly Price (£)"},
            color_discrete_sequence=["#2A9D8F"],
        )
        price_dist.update_layout(height=350, margin={"l": 40, "r": 20, "t": 40, "b": 40})
    else:
        price_dist = build_empty_figure(
            title="Price Distribution in Selected Scope",
            message="No listings found for this neighbourhood and room type.",
        )

    room_chart_df = dashboard_df[dashboard_df["neighbourhood_cleansed"] == neighbourhood].copy()

    if len(room_chart_df) > 0:
        room_type_avg = room_chart_df.groupby("room_type")["price"].mean().sort_values(ascending=False)
        room_chart = px.bar(
            x=room_type_avg.index,
            y=room_type_avg.values,
            title=f"Average Nightly Price by Room Type — {neighbourhood}",
            labels={"x": "Room Type", "y": "Average Price (£)"},
            color_discrete_sequence=["#FF6B6B"],
        )
        room_chart.update_layout(height=350, margin={"l": 40, "r": 20, "t": 40, "b": 40})
    else:
        room_chart = build_empty_figure(
            title="Average Price by Room Type (Selected Neighbourhood)",
            message="No neighbourhood-level data available.",
        )

    return prediction_title, pred_text, prediction_style, pred_note, stats_table, price_dist, room_chart


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
