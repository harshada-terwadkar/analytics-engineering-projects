# Notebook walkthrough: PayPal Fraud Detection

A beginner-friendly explanation of every section in `PayPal_Fraud_Detection.ipynb`.

---

## Section 1 — What the data can answer

Two tables are provided:

* **users** (9,944 rows): one row per user, created at sign-up. Columns: `id`, `has_email`, `phone_country`, `terms_version`, `created_date`, `country`, `birth_year`, `kyc` (identity check status), `failed_sign_in_attempts`, `is_fraud`.
* **transactions** (638,742 rows): one row per transaction. Columns: `currency`, `state`, `created_date`, `merchant_category`, `merchant_country`, `entry_method`, `user_id`, `type`, `source`, `id`, `amount_usd`, `is_crypto`.

The key observation: the fraud label is on the **user**, not the transaction. We know *who* is a fraudster, but not which specific transaction was fraudulent. This means the cleanest prediction task is:

> *After each transaction, how likely is this user to be a fraudster?*

Scoring after every transaction gives a live risk score that can trigger action before all the money is gone.

---

## Section 2 — Data audit

**Why it matters:** machine learning models silently absorb messy data. Cleaning must happen before features are built, not after.

Five issues found and addressed:

1. `amount_usd` is stored in **cents** (a coffee is 1,429 = $14.29). Divided by 100 throughout.
2. Crypto transactions (BTC, ETH, LTC) have absurd amounts — the largest is $16.5 trillion — because no currency conversion was applied. All amounts are capped at the 99.9th percentile of non-crypto transactions.
3. `merchant_category` (67%) and `merchant_country` (30%) are missing. This is structural: top-ups and bank transfers have no merchant. Missingness is left as-is and used as information.
4. 2,196 users have no transactions. They can't move money, so they are left out of the model.
5. Users who signed up in 2024 Q3 show only 0.3% fraud. This is because fraud investigations take time; that cohort simply hasn't been fully reviewed yet. No sign-up date features are included to avoid the model learning "new sign-up = safe".

---

## Section 3 — What fraudsters look like

**Profile level:**
- **Country GB** has a 6% fraud rate; France, Spain and Ireland are under 1%.
- **KYC PENDING** is 22% fraud, **KYC FAILED** is 8%, **KYC PASSED** is 4%.
- Users without any KYC (`NONE`) are 0.04% fraud — they probably can't move enough money to bother.

**Transaction behaviour:**
- Fraudsters pay in **GBP 92%** of the time vs. 47% for normal users.
- They use **bank transfers 4× more** and **ATM withdrawals 2.5× more**.
- Their transactions are **declined 2.4× more often**.
- The **MINOS source** (a top-up / transfer pipeline) handles 21% of fraudster transactions but only 1% of normal ones — a 17× lift.

**Life cycle:**
- Fraudsters **top up first** (like normal users) and then switch to cashing out between transaction 5 and 10.
- Their accounts are active for a median of **38 days**, vs. 99 days for normal users.

This tells us the model should update after every transaction: a profile score at sign-up is a reasonable start, but the big signal arrives with the first few cash-out attempts.

---

## Section 4 — Framing: real-time user scoring

**The design choice:**
- Each data point is a *(user, moment)* pair — one row per transaction, scored at the time of that transaction.
- Every feature uses only information available up to and including that transaction.
- The label is the user's `is_fraud` flag, repeated on every row.

**Avoiding leakage:**
- Features like "total account lifetime in days" are excluded because they tell the model whether the account was eventually closed.
- Running cumulative features (share of ATM transactions so far, cumulative cash-out, etc.) are safe because they only look backwards.

**Data split:** users are split 60/20/20 into train, validation and test, stratified on fraud. The same user never appears in two sets. Models are trained on users, not transactions.

**Weighting:** a user with 500 transactions would otherwise dominate training. Each user's transactions are weighted to sum to 1, so the model learns about *people*, not about the most active accounts. Fraudsters (3% of users) are up-weighted to balance the label.

---

## Section 5 — Feature engineering

**Profile features (10):** the columns known at sign-up — KYC, GB flag, email, phone-country mismatch, age, failed sign-ins, missing terms version, and KYC one-hot encoded.

**Per-transaction features:** what this transaction is (type, state, entry method, source, log-amount).

**Running history features (the most powerful):**
- `tx_n`: how many transactions the user has made so far.
- `days_since_signup`, `tx_per_day`, `tx_last_24h`, `hours_since_prev`: speed and frequency signals.
- For each category (type, state, entry method, source), the **share** of the user's history in each value.
  E.g. `share_source_MINOS` = (transactions from MINOS so far) ÷ (all transactions so far).
- `share_gbp`, `share_foreign_merchant`: same idea for currency and merchant country.
- `log_max_amount`, `log_cum_amount`: largest and total amount ever.
- `log_cum_cash_out`, `cash_out_to_top_up`: how much money has left the account vs. how much came in.

---

## Section 6 — Models and benchmarks

Four models compared on **validation users**:

| Model | What it is | User AUC | User PR-AUC |
|---|---|---|---|
| Analyst rule | Points for PENDING KYC, failed KYC, GB, MINOS, bank transfer, ATM, declined — summed | 0.872 | 0.283 |
| Profile only (logistic) | Logistic regression on sign-up features only | 0.790 | 0.110 |
| All features (logistic) | Logistic regression on all 50+ features | 0.951 | 0.639 |
| **XGBoost (real-time)** | Gradient-boosted trees on all features | **0.954** | **0.754** |

**AUC** measures whether fraudsters rank above normal users overall. **PR-AUC** (precision–recall area under curve) measures how clean the top of the ranking is. PR-AUC is more informative when fraudsters are rare (3%) because it focuses on what happens at the alert threshold, not across all possible thresholds.

XGBoost is tuned with 12 random trials, each training a different combination of `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight` and `reg_lambda`. The best trial (PR-AUC 0.754 on validation) is selected and evaluated once on the test set.

---

## Section 7 — How early can we catch them?

The model is re-evaluated using only the first n transactions per user (n = 1, 2, 3, 5, 10, 20, all). The question: can we flag fraudsters before they cash out?

**Test set results:**

| After n tx | AUC | PR-AUC |
|---|---|---|
| 1 | 0.889 | 0.370 |
| 3 | 0.942 | 0.532 |
| 5 | 0.963 | 0.652 |
| 10 | 0.983 | 0.791 |
| all | 0.990 | 0.850 |

Profile alone (sign-up features) gives AUC 0.826 from the start. Transaction behaviour adds 10+ AUC points by the 5th transaction — exactly the window before most fraudsters begin cashing out.

---

## Section 8 — Business simulation

The model is replayed on test users in time order, as if it were live:

- **Fraud loss** = all money leaving confirmed fraudsters via completed card, ATM, bank transfer or P2P transactions.
- **LOCK_USER** at score ≥ `t_lock`: that transaction and everything after is blocked.
- **ALERT_AGENT** at score ≥ `t_alert`: a human reviews within 24 hours; only outflows after the review are saved.
- **Costs:** $15 per agent review, $100 per genuinely good user wrongly locked.

The thresholds are chosen on the validation set to maximise net value (money saved minus costs), then applied once to the test set.

**Test results:**

| Policy | Fraudsters caught | Fraud saved | Agent reviews | Good users locked | Net value |
|---|---|---|---|---|---|
| No model | 0% | $0 | 0 | 0 | $0 |
| Alert only | 100% | $319,411 | 651 | 0 | $309,646 |
| Lock only | 83% | $290,357 | 62 | 12 | $288,227 |
| **Alert + lock (chosen)** | **100%** | **$332,572** | **651** | **12** | **$321,607** |

The two-tier policy wins: most flagged users get an alert (cheap, no disruption to genuine users), and the small group with very high scores gets locked immediately (saves a few extra days of outflows before the agent reviews).

---

## Section 9 — Operational policy: `patrol(user_id)`

```python
def patrol(user_id):
    """Suggested actions based on the user's profile and all transactions so far."""
    if user_id not in LATEST.index:
        return []                          # no transactions -> no money at risk
    risk = model.predict_proba(F.loc[[LATEST[user_id]]])[:, 1][0]
    if risk >= T_LOCK:
        return ['LOCK_USER', 'ALERT_AGENT']
    if risk >= T_ALERT:
        return ['ALERT_AGENT']
    return []
```

Called after every new transaction. The thresholds (`T_ALERT = 0.2`, `T_LOCK = 0.97`) come from the business simulation in section 8 and can be updated without changing the model.

---

## Section 10 — Summary and limitations

**What the model delivers:**
- 0.990 AUC, 0.850 PR-AUC on unseen users.
- Catches 100% of test fraudsters with the chosen thresholds.
- Prevents 94% of fraud loss ($332k of $355k).
- Wrongly locks 12 out of 1,490 good users (0.8%).

**Limitations:**
- Small test set (~60 fraudsters). Use cross-validation by user for production estimates.
- No fraud timestamp: "money saved" assumes all outflows would have been lost without the model. That's an upper bound.
- Cost assumptions are placeholders.
- Recent cohorts are under-labelled. Retrain on matured cohorts in production.

**Next steps:**
- Device and IP signals.
- Network features: who sends to whom via P2P transfers.
- Retrain pipeline on rolling window.
- A/B test of the live policy.
