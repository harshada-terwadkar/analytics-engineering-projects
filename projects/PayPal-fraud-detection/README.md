# 🛡️ PayPal Fraud Detection

Scoring every user after each transaction to catch fraudsters as early as possible — before they drain the account — and deciding the right action: lock immediately, send to a human agent, or do nothing.

![Fraud lifecycle](images/fraud_lifecycle.png)

> *Fraudsters top up normally, then start cashing out between transactions 5 and 10. The model flags them before that window closes.*

---

## Business problem

PayPal faces fraudsters who fund a new account, drain money via ATM withdrawals and bank transfers, then disappear. Every flag triggers a manual review, and every wrong lock disturbs a genuine customer. The goal is to catch fraud early, spend as few review hours as possible, and almost never lock a good user without warning.

## Results

| Metric | Result |
|---|---|
| User-level AUC (test) | **0.990** |
| User-level PR-AUC (test) | **0.850** |
| Fraudsters caught | **100%** with chosen thresholds |
| Fraud loss prevented | **$332,572 of $355,392 (94%)** |
| Good users wrongly locked | **12 of 1,490 (0.8%)** |
| Agent reviews triggered | 651 for 1,550 test users |
| **Net value vs. no model** | **+$321,607** |

Detection quality by transactions seen — the model is already actionable before fraudsters cash out:

| After n transactions | AUC | PR-AUC |
|---|---|---|
| 1 | 0.889 | 0.370 |
| 3 | 0.942 | 0.532 |
| 5 | 0.963 | 0.652 |
| 10 | 0.983 | 0.791 |
| all | 0.990 | 0.850 |

## Key findings

- **GB accounts are 6× riskier.** Most of Europe is under 1% fraud rate; GB is 6%. KYC PENDING is 22%.
- **The MINOS pipeline is a 17× signal.** It handles 21% of fraudster transactions but only 1% of normal ones.
- **GBP usage and ATM withdrawals are the tell.** Fraudsters transact in GBP 92% of the time; they use ATMs 2.5× more than normal users.
- **Fast life cycle.** Accounts last a median of 38 days vs. 99 for normal users — there is a hard deadline on catching them.

![Feature importance](images/feature_importance.png)

---

## How I built it

**Real-time user scoring, not transaction classification.** Fraud is labelled per user, not per transaction. I score the user after every new transaction using all their history up to that point — no future leakage. This mirrors what a live system would do.

**Weighted training by user, not transaction.** A user with 500 transactions would otherwise dominate the loss. Each user's transactions are weighted to sum to 1, so the model learns about people, not active accounts.

**Running share features, not raw counts.** `share_source_MINOS` (MINOS transactions ÷ all transactions so far) captures behavioural drift. Raw counts grow with activity and would encode account age instead of behaviour.

**No account lifetime features.** Knowing how long an account has been open tells the model whether it was eventually closed — that is leakage. Time-elapsed features stay out.

**Benchmarked four models.** Analyst rule → profile-only logistic regression → all-features logistic regression → XGBoost. Each step is justified by the PR-AUC jump, not assumed. Logistic regression with all features (0.639 PR-AUC) versus XGBoost (0.754) shows the non-linear interactions are real.

**Two-tier policy, not a single threshold.** One threshold locks (high confidence, act immediately). A lower threshold alerts a human agent to review. The two-tier approach saves 12 extra days of outflows that an alert-only policy would miss.

## Policy

| Score | Action | Why |
|---|---|---|
| ≥ 0.97 | **LOCK + ALERT** | Very high confidence: block immediately; agent unlocks fast if wrong |
| ≥ 0.2 | **ALERT** | Suspicious but uncertain: let through, agent reviews |
| < 0.2 | — | Keep scoring on every new transaction |

| Early detection curve | Precision / recall at threshold |
|---|---|
| ![Early detection](images/early_detection.png) | ![PR threshold](images/precision_recall_threshold.png) |

---

## Limitations and next steps

- ~60 test fraudsters is a small sample; metrics can shift 2–3 points across splits. k-fold by user gives tighter estimates.
- No fraud timestamp means "money saved" is an upper bound — we assume all outflows would have been lost.
- Review cost ($15) and wrong-lock cost ($100) are placeholders. The policy grid re-optimises with real numbers.
- Next: device and IP signals, P2P transfer network features, and a live A/B test of the patrol policy.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Jupyter
