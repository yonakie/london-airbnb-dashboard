# London Airbnb Price Predictor Dashboard

An interactive Dash application for predicting nightly Airbnb prices in London based on neighbourhood, room type, and property characteristics.

## Features

- **Interactive Filters** — Neighbourhood, room type, property type, bedrooms, and guest capacity selectors
- **Price Prediction** — ML-powered price estimates with confidence intervals
- **Neighbourhood Statistics** — Summary metrics including average/median price and guest reviews
- **Price Distribution Chart** — Visualize price spread in selected neighbourhood
- **Room Type Comparison** — Compare pricing across room types within a neighbourhood

## Tech Stack

- **Frontend:** Dash (Python) + Plotly
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Deployment:** Gunicorn + Render

## Local Setup

### Prerequisites
- Python 3.9+
- `listings.csv.gz` (Inside Airbnb London dataset)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yonakie/london-airbnb-dashboard.git
   cd london-airbnb-dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place `listings.csv.gz` in the project root directory

5. Run the app:
   ```bash
   python app.py
   ```

6. Open [http://localhost:8050](http://localhost:8050) in your browser


## File Structure

```
london-airbnb-dashboard/
├── app.py                 # Main Dash application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Dataset

The application uses Inside Airbnb London data (`listings.csv.gz`):
- **Source:** https://insideairbnb.com/get-the-data
- **Size:** ~62,000 listings

## Model Performance

The dashboard currently uses a tuned XGBoost regressor trained on the 99th-percentile-capped dataset:
- **RMSE (3-fold CV):** ~88.62/night
- **R² Score:** ~0.65
- **Validation Strategy:** 3-fold cross-validation on capped data

LightGBM remains available as a backup artifact for side-by-side comparison.

<!-- ## Future Improvements

- [ ] Season/month-based pricing adjustments
- [ ] Calendar availability integration
- [ ] Host performance tier predictions
- [ ] Amenity-specific pricing breakdown
- [ ] Geographic mapping (Folium integration) -->

