# The Flight Price Notebook, Explained

This guide walks through [Google_Flights_Airfare_Forecast.ipynb](../Google_Flights_Airfare_Forecast.ipynb) section by section. For each step it covers **what** the code does, **why** it was done that way, and how it serves the **use case**.

---

## The use case in one paragraph

Google Flights wants an **Interactive Calendar**. A traveller picks a route, airline, fare class and departure date, and the calendar shows the predicted price for each day they could book, for example the next 7 days. It highlights the cheapest day and how much they'd save compared with booking today. Building it takes two things:

1. **A price model**: given a booking date and the flight details, predict the ticket price.
2. **A scenario ("what-if") tool**: run the model over many candidate booking dates and compare the results.

Everything in the notebook leads up to those two pieces.

---

## Setup cell

**What:** imports the libraries, fixes a random seed (`SEED = 123`), and sets a chart style.

| Library | Used for |
|---|---|
| `pandas`, `numpy` | loading and manipulating the table of bookings |
| `matplotlib` | charts |
| `scikit-learn` | label encoding, random forest (feature selection), metrics, random hyperparameter search |
| `xgboost` | the main prediction model |

**Why:** the seed makes results reproducible, so you get the same numbers every time you run it. The chart style (one fixed colour per fare class, light gridlines) keeps every chart consistent and easy to read.

---

## Step 1: Problem definition and loading data

**What:** `pd.read_csv("google_flights_airfare_data.csv")` loads 225,209 bookings. Each row is one ticket sold:

`transaction_id, flight_id, airline, origin, destination, booking_dt, departure_dt, price, fare_class`

**Why:** this is the raw history the model learns from. Each row says "someone booked this flight on this date and paid this much".

---

## Step 2: Exploratory Data Analysis (EDA)

The goal is to understand the data before modelling, so we know which features matter and what problems to fix.

### Overview
**What:** counts rows, unique flights and categories, and the date ranges.
**Found:** 3 airlines, 5 origins, 5 destinations (25 routes), and 3 fare classes. Departures run from April 2022 to September 2024, booked 1 to 179 days ahead.
**Why it matters:** it tells us which routes the tool can support. JFK to LAX isn't in the data, so the example uses JFK to MIA.

### Missingness
**What:** checks for missing values, duplicates, negative prices, and bookings made after departure.
**Found:** the data is clean, apart from one duplicated transaction ID and one reused flight ID. Both are harmless because the model never uses IDs.
**Why:** bad data creates bad predictions, so we check before trusting it.

### Univariate analysis
**What:** bar charts of bookings per airline, fare class and origin, a histogram of days-before-departure, and bookings per month.
**Why:** this shows whether any group is so rare that the model can't learn it. Here the groups are balanced.

### Target analysis (the most important part of EDA)
"Target" means the thing we predict, which is **price**.

- **Price distribution is right-skewed**: most tickets are cheap and a few are very expensive. *That's why the model predicts `log(price)`* (see Step 6).
- **Price vs days before departure**: prices are flat from 180 to about 60 days out, then roughly **double** inside 30 days. *This is the core insight behind the feature.*
- **Seasonality**: Feb to Jun departures are the most expensive and Aug to Nov the cheapest.
- **Booking weekday**: booking on Fri or Sat costs about 15 to 20% more than on Mon or Tue. *This is what makes a 7-day calendar useful: the price changes within a week.*
- **Departure hour**: early morning flights cost the most.
- **Destination**: Miami is the most expensive destination and Boston the cheapest.
- **Noise**: the same flight in the same week can vary about 42% in price. *That's why we judge the model on average prices too, not only single tickets.*
- **Oddity**: Business is cheaper than Economy. That's unrealistic and probably a quirk of the simulation, so we flag it and keep the data as it is.

---

## Step 3: Data preparation

### Format fields
**What:**
- Converts `booking_dt` and `departure_dt` from text to real datetimes, so we can subtract them and pull out the weekday, month and so on.
- Converts text categories (airline, origin, destination, fare class) into numbers with `LabelEncoder`, e.g. Economy becomes 1. The encoders are saved so the scenario tool can encode a user's choices the same way.

**Why:** ML models only understand numbers.

### Split data (chronological)
**What:**
- **Train:** flights departing before Oct 2023 (62%)
- **Validation:** Oct 2023 to Feb 2024 (21%), used to tune settings
- **Test:** Mar to Sep 2024 (17%), used only for the final score

**Why chronological and not random?** In real life the model always predicts **future** flights. A random split would let the model "peek" at flights departing on the same days it is tested on, which gives scores that are too optimistic. Splitting by time mimics real use honestly.

---

## Step 4: Feature engineering

"Features" are the inputs the model uses. The function `add_features()` builds them, and the **same function is reused by the scenario tool**, so training and prediction always see identical inputs.

### Compute delta
- `days_before` = departure minus booking, in days (with decimals). **This is the single most important feature.**
- `weeks_before` and `last_minute` (14 days or less) are simpler versions of the same idea.

### Decompose timestamps
Breaks each date into parts the model can learn patterns from:

| From booking date | From departure date |
|---|---|
| `book_dow` (weekday), `book_weekend` (Fri/Sat), `book_hour`, `book_month`, `book_day` | `dep_year`, `dep_month`, `dep_day`, `dep_dow`, `dep_doy` (day of year, for seasonality), `dep_hour`, `dep_weekend` |

**Why:** a raw timestamp like "2024-11-04 09:13" means nothing to a model. "It's a Monday, 27 days before departure, flying at 8am in December" is what actually drives price.

---

## Step 5: Feature selection

### Random forest feature importance
**What:** trains a quick random forest on a 40k-row sample and asks which inputs it relied on most.
**Result:** fare class and days-before dominate, then destination, departure hour and day of year.
**Why a random forest?** It's fast, handles non-linear patterns, and gives a built-in importance score. That makes it a standard way to rank features.

### Filter columns
**What:** keeps features above 0.5% importance, and always keeps the inputs a user selects in the calendar (route, airline, fare class, booking weekday, days-before). It drops `dep_year`.
**Why drop the year?** Tree models can't extrapolate. They've never seen 2025, so a year feature would be useless or misleading for future dates. Dropping weak features also reduces overfitting and speeds up training.

---

## Step 6: Model selection

### Baseline model (XGBoost)
**What:** an XGBoost regressor trained to predict `log(price)`.

**Why XGBoost?**
- It's the industry standard for tabular data (rows and columns).
- It captures interactions automatically, e.g. "last-minute **and** First class **and** Miami is very expensive".
- It's fast on 200k rows and needs little preprocessing.

**Why predict log(price)?** Prices are skewed and pricing effects are multiplicative ("First class costs about 2x", "last-minute costs about 2x"). In log space those multiplications become additions, which trees model more easily, and a few huge fares don't dominate training.

### Hyperparameter tuning
**What:** `ParameterSampler` tries 15 random combinations of settings (number of trees, depth, learning rate, etc.), scores each on the **validation** set, and keeps the best (lowest RMSE).
**Why random search?** It finds good settings much faster than trying every combination (grid search).

Then three more things happen:
- **Refit on train plus validation**, so the final model learns from the most recent prices.
- **Smearing factor (1.129):** a log-model predicts the *typical* (median) price. Multiplying by this factor converts it to the *average* price, which is right for dollar savings.
- **Quantile model:** a second XGBoost predicts the 25th and 75th percentile prices. That gives the "typical range" band in the calendar, which is honest given how noisy prices are.

---

## Step 7: Evaluation

### Metrics used
| Metric | Meaning in plain words |
|---|---|
| **MAPE** | average % error ("off by X% on average") |
| **MAE** | average dollar error |
| **RMSE** | dollar error that punishes big misses more |
| **R²** | share of price variation explained (1.0 is perfect, 0 is no better than guessing the average) |

### Business method (benchmark)
**What:** the "no ML" rule an analyst might use: predict the **median historical price for the same route, fare class and weeks before departure**.
**Why:** a model is only worth building if it beats the simple approach. This gives the comparison context.

### Results on the 2024 test flights
| | MAPE | R² |
|---|---|---|
| Business rule (single tickets) | 53% | 0.38 |
| **XGBoost (single tickets)** | **51%** | **0.50** |
| Business rule (weekly averages) | 25% | 0.51 |
| **XGBoost (weekly averages)** | **20%** | **0.77** |

### Simpson's paradox check
**What:** repeats the comparison inside each fare class and each lead-time bucket.
**Why:** a model can win overall but lose in every subgroup (Simpson's paradox). Here XGBoost wins in **every** subgroup, so the improvement is real.

### Why single-ticket MAPE is about 50%
The data itself is noisy: the same flight, same week, can be $660 or $1,190. No model can predict that random part. What the calendar needs is the **expected** price for a day, and at that level the model is accurate (R² 0.77). The chart "Economy, test period" shows the predicted curve sitting right on the actual one.

---

## Step 8: Airfare scenario planning (the feature itself)

### `predict_booking_calendar(...)`
**Inputs:** today's date, departure date, airline, origin, destination, fare class, and window length (default 7 days).
**What it does:**
1. Lists every candidate booking day from today to today plus 6, never on or after departure.
2. For each day, creates booking times from 8am to 8pm (to average out time-of-day noise).
3. Builds features with the **same** `add_features()` and the saved encoders.
4. Predicts the expected price and the 25 to 75% range, then averages per day.
5. Returns a table with each day's price, weekday, days before departure, and the difference vs today.

### `plot_booking_calendar(...)`
Draws the line of expected prices with the shaded typical range, and highlights the cheapest day with the saving.

### `booking_advice(...)`
Turns the table into a sentence, for example:
> "Wait and book on Monday Nov 04 for about $904, saving ~$69 (7%) vs booking today. Avoid Thursday Nov 07 (~$1,039)."
If no day is cheaper, it says to book today.

### Full calendar chart
Shows predicted prices from 82 days out until departure. You can see the weekly Fri/Sat spikes (orange dots) and the steep climb after the 60-day mark. That's the visual story for the presentation.

### More example scenarios
Runs five different routes and fare classes and shows the best day and saving for each. These are useful as presentation examples.

### Backtest: do users actually save?
**What:** replays about 3,200 real 2024 searches. For each one it picks the day the model says is cheapest in the next 7 days, then checks the **actual** price on that day.
**Result:** following the calendar saves about **$31 (5.6%)** on average vs booking immediately, and the chosen day beats a random day 59% of the time.
**Why:** this is the business proof. It measures real savings, not just model accuracy.

---

## How it all fits together

```
Raw bookings ──► Clean & format ──► Features (days_before, weekday, season...)
                                          │
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
          Train XGBoost (log price)              Quantile XGBoost (range)
                       │                                     │
                       └───────────► Scenario tool ◄─────────┘
                                         │
              User picks route + dates ──┘──► Price per booking day
                                              Best day + saving
```

## Key takeaways for travellers (and for the presentation)
1. **Book 60 or more days ahead.** Inside 30 days, fares roughly double.
2. **Book Monday to Wednesday, not Friday or Saturday** (about 15 to 20% cheaper).
3. **Close to departure, book now.** Waiting usually costs more, and the calendar tells you so.
4. The calendar saves the average user about **5 to 6%** per ticket, more on far-out trips with flexible booking days.
