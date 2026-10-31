# ✈️ Google Flights Airfare Forecast

Predicting airline ticket prices and building a **best day to book** calendar — given a route, airline, fare class and departure date, it tells you when to book over the next 7 days and how much you save versus booking today.

![Booking calendar for JFK to MIA](images/booking_calendar_jfk_mia.png)

> *Searching on Nov 1 for a Dec 1 Delta flight from JFK to Miami: wait until Monday Nov 4, save **$69 (7%)**.*

---

## Business problem

Google Flights users ask: *when is the best time to book?* The goal is a data-driven calendar that answers that for any route and fare class. Two things are needed — a price model that predicts the fare for a given booking date, and a scenario tool that runs it over a window of dates, finds the cheapest day, and shows the saving.

## Results

| Metric | Result |
|---|---|
| Weekly average accuracy (2024 test flights) | **R² 0.77**, MAPE 20% vs. 25% for the business rule |
| Single-ticket accuracy | R² 0.50, MAPE 51% (prices are inherently noisy — ±42% within the same week) |
| Backtest: average saving from following the calendar | **$31 per ticket (5.6%)** across 3,230 real 2024 searches |
| Calendar beats a random day in the same week | **59%** of the time |

## Key insights

- **Book 60+ days ahead.** Prices are flat until ~60 days out, then roughly double inside 30 days.
- **Avoid Friday and Saturday bookings.** Within any week, Mon–Wed is 15–20% cheaper.
- **Seasonality is real.** Feb–Jun departures cost the most; Aug–Nov the least.
- **Close to departure, book today.** The calendar says so — waiting usually costs more.

![Price vs days before departure](images/price_vs_days_before.png)

---

## How I built it

**Chronological split, not random.** Train before Oct 2023, validate Oct 2023–Feb 2024, test Mar–Sep 2024. A random split would let the model peek at future flights — it would score well in training and fail in production.

**XGBoost on log(price).** Pricing effects are multiplicative ("First class costs 2×", "last-minute costs 2×"), so log space is the right domain. A smearing correction (×1.129) converts the model's median prediction to the mean price used in dollar savings.

**One `add_features()` function shared by training and the scenario tool.** If they diverge, the calendar silently breaks. Keeping a single function makes that impossible.

**Quantile models for the price range.** A single expected price is honest but misleading — the same flight can vary 42% in a week. The 25th–75th percentile band in the calendar shows users what "typical" means for their booking.

**Dropped `dep_year`.** Tree models cannot extrapolate to years they have never seen. Keeping the year feature would make predictions for 2025 meaningless.

**Benchmarked against a business rule.** The "no-ML" baseline: median historical price for the same route, fare class and lead-time bucket. XGBoost beats it in every fare class and every lead-time bucket — no Simpson's paradox hiding a bad subgroup.

| Predicted vs actual | Feature importance |
|---|---|
| ![Predicted vs actual](images/predicted_vs_actual.png) | ![Feature importance](images/feature_importance.png) |

---

## Limitations and next steps

- Data is simulated. Business fares cheaper than Economy is a known quirk — confirm with the data owner before any real launch.
- The calendar shows expected prices, not guarantees. Single-ticket noise is irreducible.
- Model is trained to Sep 2024; seasonal patterns are assumed to hold beyond that.
- Next: retrain on rolling fresh data, add holiday and event features, A/B test the calendar on booking conversion.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Jupyter
