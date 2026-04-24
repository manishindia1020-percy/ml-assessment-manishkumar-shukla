# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

# B1. Problem Formulation

## (a) ML Problem Formulation

**Target Variable:** `items_sold` — the number of items sold at a given store in a given month under a specific promotion.

**Candidate Input Features:**

| Feature | Source Table | Description |
|---|---|---|
| `promotion_type` | Promotions | One of 5 promotion types |
| `store_size` | Store Attributes | Small / Medium / Large |
| `location_type` | Store Attributes | Urban / Semi-urban / Rural |
| `monthly_footfall` | Store Attributes | Average customer visits per month |
| `competition_density` | Store Attributes | Number of competing stores nearby |
| `month` | Calendar | Captures seasonality |
| `is_weekend` | Calendar | Binary flag |
| `is_festival` | Calendar | Binary flag for festival periods |
| `customer_demographics` | Store Attributes | Age group, income band |
| `past_items_sold` | Transactions | Lagged sales (previous month) |
| `promotion_history` | Transactions | Which promotion ran last month |

**Type of ML Problem:** Supervised Regression.

**Justification:** The target variable `items_sold` is a continuous numeric quantity — there is no natural category boundary between "good" and "bad" sales. We want to predict the exact volume of items sold under each promotion scenario so the marketing team can select the promotion that maximises that value. This is a regression problem, not classification. Specifically, because we want to *compare* the predicted output across five promotion options for each store-month, this can also be framed as a **counterfactual prediction** task: "for Store 12 in December, what would items_sold be under each of the 5 promotions?" The recommendation is then simply the promotion with the highest predicted value.

---

## (b) Why Items Sold is a Better Target Than Revenue

Revenue is the product of items sold and price per item. When promotions involve deep discounts (e.g. a 50% Flat Discount), revenue per unit drops significantly even when customer demand is high. Consider this example:

| Promotion | Items Sold | Avg Price | Revenue |
|---|---|---|---|
| Flat Discount (50% off) | 200 | £10 | £2,000 |
| Loyalty Points Bonus | 120 | £20 | £2,400 |

A model trained to maximise revenue would recommend Loyalty Points Bonus here. But the business objective is to move stock and drive footfall — 200 items sold is a far better outcome. Using revenue as the target would cause the model to systematically penalise high-volume, discount-driven promotions, directly contradicting the business goal.

**Broader Principle — Target Variable Alignment:**

The target variable must directly measure the business outcome you are trying to optimise, not a proxy that is distorted by factors outside your control. In this case, the discount rate (part of the promotion itself) is an input feature, not an outcome. Using a target that is contaminated by input features (price is set by the promotion choice) introduces a structural confound: the model would learn that cheaper prices produce lower revenue, not that they produce higher volume. The principle is — *choose a target that is causally downstream of your decisions, not one that is algebraically entangled with your inputs.*

---

## (c) Better Modelling Strategy Than One Global Model

A single global model trained across all 50 stores assumes the same feature-response relationship holds everywhere. In reality, a Flat Discount may drive strong volume in a price-sensitive rural store but have little additional effect in an affluent urban store where customers already have high willingness to pay.

**Proposed Strategy: Location-Type Stratified Models**

Train one model per `location_type` group — Urban, Semi-urban, and Rural. Each model learns the promotion-response patterns specific to that store environment. With 50 stores across 3 location types and 36 months of data, each group will have approximately 600 store-month training rows — sufficient for a reliable model.

```
Urban stores     → Model A
Semi-urban stores → Model B
Rural stores     → Model C
```

At prediction time, each store is routed to its corresponding model.

**Alternative: Mixed-Effects (Hierarchical) Model**

A single model with `store_id` as a random effect (using `statsmodels MixedLM` or `pymer4`). Fixed effects capture the global pattern shared across all stores (promotion impact, seasonality, footfall); random effects capture store-specific deviations from that baseline. This is statistically principled and avoids the sample-size problem that arises when some location groups are small.

**Why Not One Model Per Store?**

With only ~36 months per individual store, a fully separate model per store would have too little training data, would overfit to idiosyncratic events in that store's history, and would be impossible to maintain (50 separate retraining pipelines). Grouping by location type is the right balance between specificity and statistical power.

---

# B2. Data and EDA Strategy

## (a) Data Joining and Grain

**The Four Source Tables:**

| Table | Key Columns |
|---|---|
| Transactions | `store_id`, `transaction_date`, `items_sold`, `promotion_id` |
| Store Attributes | `store_id`, `store_size`, `location_type`, `footfall`, `competition_density` |
| Promotion Details | `promotion_id`, `promotion_type`, `discount_depth` |
| Calendar | `date`, `is_weekend`, `is_festival`, `month`, `year` |

**Join Strategy:**

```
Transactions
    JOIN Store Attributes  ON store_id
    JOIN Promotion Details ON promotion_id
    JOIN Calendar          ON transaction_date = date
```

All joins are left joins anchored on the Transactions table to preserve every transaction row. Store attributes and promotion details are static lookups. Calendar is a date-level lookup.

**Grain of the Final Modelling Dataset:**

One row = **one store × one month × one promotion type.**

This is the unit at which the business makes decisions: "which promotion should Store 12 run in December?"

**Aggregations Performed Before Modelling:**

| Aggregation | Logic |
|---|---|
| `items_sold` | SUM of items sold across all transactions for that store-month-promotion |
| `avg_basket_size` | MEAN of transaction-level basket sizes |
| `total_footfall` | SUM or provided at store-month level from store attributes |
| `promotion_type` | The promotion that ran (assumed one per store per month; if multiple ran, take the primary one by revenue share) |
| `competition_density` | Store-level attribute — constant per store, joined directly |
| `is_festival` | OR across all days in the month (1 if any day in the month was a festival) |

**Edge Cases to Handle:**
- Stores with no transactions in a month should appear as a row with `items_sold = 0`, not be silently dropped.
- If a store ran two promotions in a month, create two rows (one per promotion) with proportionally attributed sales, or flag and exclude from training.

---

## (b) EDA Plan

| # | Analysis | What to Look For | How it Influences Modelling |
|---|---|---|---|
| 1 | **Promotion vs Average Items Sold (bar chart)** | Which promotion type produces the highest mean sales across all stores | If one promotion dominates, check for selection bias — was it only run in high-footfall stores? This affects feature importance interpretation |
| 2 | **Promotion × Location Type Interaction (grouped bar chart)** | Whether Flat Discount works better in Rural vs Urban | Confirms the need for segmented models; informs whether `promotion_type × location_type` interaction terms should be engineered |
| 3 | **Monthly Sales Trend (line chart, all stores averaged)** | Seasonal peaks — December, festival months | Justifies including `month` and `is_festival` as features; may require month-of-year sine/cosine encoding for cyclicality |
| 4 | **Correlation Heatmap (numeric features vs items_sold)** | Which numeric features (footfall, competition_density, basket_size) correlate most strongly with items_sold | High-correlation features are prioritised; near-zero correlations may be dropped; multicollinearity between footfall and store_size flagged |
| 5 | **Promotion Frequency Distribution (countplot)** | Whether some promotions are run far more often than others (imbalance) | If 80% of rows have no promotion, the model will be biased — requires resampling or weighted loss (see B2c) |
| 6 | **Residual plot of a simple baseline model** | Whether errors are random or structured (e.g. all large errors in December or in rural stores) | Reveals whether additional features or interaction terms are needed; structured residuals signal a missing predictor |

---

## (c) Addressing the Promotion Imbalance (80% No-Promotion)

**How it affects the model:**

When 80% of training rows have no active promotion, the model will be predominantly trained on non-promotional behaviour. It will underestimate the uplift from promotions because it has seen far fewer examples of promoted sales. In particular, it will default toward predicting the no-promotion baseline and will fail to correctly distinguish between the five promotion types due to sparse training signal for each.

**Steps to address it:**

1. **Weighted training loss:** Assign higher sample weights to promoted rows during model training. In scikit-learn this is `RandomForestRegressor.fit(X_train, y_train, sample_weight=weights)` where `weights` is higher for promoted rows. This causes the model to pay more attention to promotional examples without discarding non-promotional data.

2. **Stratified subsampling:** Downsample non-promotional rows in the training set so the promoted/non-promoted ratio is closer to 50:50. Evaluate whether this improves prediction accuracy on promoted test rows specifically.

3. **Separate models:** Train one model on non-promotional data (to predict baseline demand) and a separate model on promotional data (to predict uplift). The final recommendation is: `baseline_prediction + uplift_prediction`. This cleanly separates the two regimes and avoids the imbalance problem entirely.

4. **Analyse separately:** Even if keeping one model, always report evaluation metrics separately for promoted vs non-promoted test rows. A model that performs well overall but poorly on promoted rows is useless for this business problem.

Note: standard classification resampling techniques like SMOTE are not appropriate here because this is a regression problem. Oversampling for regression requires techniques such as SMOGN or simply the sample-weighting approach described above.

---

# B3. Model Evaluation and Deployment

## (a) Train-Test Split Strategy and Metrics

**Why a Random Split is Inappropriate:**

The dataset has explicit temporal structure — each row belongs to a specific month. A random split would scatter future months (e.g. December 2024) into the training set and past months (e.g. January 2022) into the test set. The model would effectively be trained on future data and evaluated on past data. This produces evaluation metrics that are far too optimistic because the model has already "seen" the market conditions of the period it is being tested on. In deployment, the model only ever sees past data and predicts the future — the evaluation setup must reflect this.

**Concrete Split for This Dataset (3 years × 50 stores = ~1,800 store-month rows):**

```
Training set : Jan 2022 – Jun 2024  (30 months × 50 stores = ~1,500 rows)
Test set     : Jul 2024 – Dec 2024  ( 6 months × 50 stores =   ~300 rows)
```

This simulates real deployment: train on all available history, evaluate on the most recent 6 months — the period most representative of conditions at the time of deployment.

**Preferred Approach: Walk-Forward Cross-Validation**

Rather than a single split, use an expanding-window cross-validation:

```
Fold 1: Train Jan–Dec 2022        → Test Jan 2023
Fold 2: Train Jan 2022–Jan 2023   → Test Feb 2023
Fold 3: Train Jan 2022–Feb 2023   → Test Mar 2023
... and so on through Dec 2024
```

This produces ~24 evaluation windows and a robust, low-variance estimate of model performance. It also shows whether performance is stable over time or degrading, which is a signal that the model needs more frequent retraining.

**Evaluation Metrics and Business Interpretation:**

| Metric | Formula | Business Meaning |
|---|---|---|
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors heavily. An RMSE of 15 items/store/month means the model's largest errors are roughly 30 items. At £5/item that represents ~£150 in misallocated promotional spend per store per month. |
| **MAE** | mean(|y − ŷ|) | Average absolute error in items_sold units. Easier to communicate: "on average, our recommendation is off by X items per store per month." Preferred for stakeholder reporting. |
| **MAPE** | mean(|y − ŷ| / y) | Percentage error — essential for comparing accuracy across stores of very different sizes. A large urban store selling 500 items and a small rural store selling 50 items should not be evaluated on the same absolute scale. |
| **Promotion-level MAE** | MAE computed per promotion type | The most operationally important metric. If MAE for Flat Discount is 5 but MAE for BOGO is 40, the recommendation engine cannot be trusted for BOGO decisions specifically. |

---

## (b) Explaining Model Decisions Using Feature Importance

**Why Global Feature Importance is Not Enough:**

The model's global `feature_importances_` (from a Random Forest) tells us which features matter most *on average across all predictions*. But to explain why Store 12 gets a different recommendation in December vs March, we need **local, per-prediction explainability**.

**Recommended Approach: SHAP Values (SHapley Additive exPlanations)**

SHAP decomposes each individual prediction into the additive contribution of each feature for that specific data point. The sum of all SHAP values equals the difference between that prediction and the model's baseline average prediction.

**Example SHAP breakdown for Store 12:**

| Feature | Dec value | SHAP (Dec) | Mar value | SHAP (Mar) |
|---|---|---|---|---|
| `month` | 12 | +22 items | 3 | −8 items |
| `is_festival` | 1 | +15 items | 0 | 0 |
| `promotion_Loyalty_Points` | 1 | +10 items | 0 | — |
| `promotion_Flat_Discount` | 0 | — | 1 | +18 items |
| `footfall` | 420 | +6 items | 310 | −4 items |
| `competition_density` | 3 | −4 items | 3 | −4 items |
| **Prediction** | | **~149 items** | | **~102 items** |

**How to communicate this to the marketing team:**

> "In December, Store 12 benefits from peak seasonal demand (month=12, +22 items) and a local festival (+15 items). Under these conditions, customers are already motivated to buy — the Loyalty Points Bonus captures their intent and adds incremental basket size without the cost of a discount. In March, underlying demand is lower (month=3, −8 items) and there is no festival. A Flat Discount is recommended because it provides a direct price incentive that stimulates purchases that would not otherwise occur. The model has learned this pattern from historical data across all semi-urban stores with similar footfall profiles."

This framing translates model logic into actionable business narrative without requiring the marketing team to understand how a Random Forest works.

**Implementation:**
```python
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Waterfall plot for a single store-month prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[store12_dec_index],
    base_values=explainer.expected_value,
    data=X_test.iloc[store12_dec_index],
    feature_names=feature_names
))
```

---

## (c) End-to-End Deployment Pipeline

### Step 1 — Save the Trained Model and Pipeline

```python
import joblib
from datetime import date

version = date.today().strftime("%Y%m%d")
joblib.dump(full_pipeline, f'models/promotion_model_{version}.pkl')
joblib.dump(preprocessor,  f'models/preprocessor_{version}.pkl')
```

Save with a date-stamped version tag. Maintain a `model_registry.json` that records which version is currently live in production, enabling instant rollback.

### Step 2 — Monthly Data Preparation

At the start of each month, an automated script:

1. Queries the four source tables for the upcoming month's store and calendar data
2. Aggregates to store × month grain (identical logic to training — this must be a shared, versioned function, not duplicated code)
3. Applies the **same preprocessing pipeline** (saved in Step 1) to transform the new data
4. Runs a data validation check before prediction:
   - No stores are missing from the input
   - All categorical values are within the known training set vocabulary
   - No null values in required columns
   - Feature distributions are not dramatically shifted (see monitoring below)

### Step 3 — Generating Recommendations

```python
pipeline = joblib.load('models/promotion_model_latest.pkl')

# For each store, score all 5 promotion options
recommendations = []
for store_id in store_list:
    store_preds = {}
    for promo in ['flat_discount', 'bogo', 'free_gift', 'category_offer', 'loyalty_points']:
        row = build_feature_row(store_id, upcoming_month, promo)
        store_preds[promo] = pipeline.predict(row)[0]
    best_promo = max(store_preds, key=store_preds.get)
    recommendations.append({'store_id': store_id, 'recommended_promotion': best_promo,
                             'predicted_items_sold': store_preds[best_promo]})

output_df = pd.DataFrame(recommendations)
output_df.to_csv(f'recommendations/rec_{upcoming_month}.csv', index=False)
```

### Step 4 — Automation

Schedule using **Apache Airflow** with a monthly DAG:

```
[Data Extract] → [Data Validation] → [Feature Engineering] → [Score All Promos]
                                                                       ↓
                                              [Output CSV] → [Email/Slack to Marketing]
```

### Step 5 — Performance Monitoring

| Signal | Method | Alert Threshold |
|---|---|---|
| **Prediction accuracy** | Rolling 3-month MAE on actuals vs predictions (once actuals arrive) | Alert if MAE increases >20% above training MAE |
| **Feature distribution drift** | Population Stability Index (PSI) on key features: `month`, `footfall`, `competition_density` | PSI > 0.1: monitor. PSI > 0.2: investigate. PSI > 0.25: pause and retrain |
| **Prediction distribution drift** | Track mean and std of predicted items_sold each month | Flag if monthly mean shifts >2 standard deviations from historical mean |
| **Business KPI alignment** | Compare actual items_sold for recommended-promotion stores vs holdout control group | Reviewed in monthly business meeting |

**PSI formula:**
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)

### Step 6 — Retraining Triggers and Process

Retrain when any of the following conditions are met:

1. Rolling MAE on live data exceeds training MAE by >20% for 2 consecutive months
2. A new promotion type is introduced (model has never seen it in training)
3. PSI on any key feature exceeds 0.25 (distribution has shifted significantly)
4. Scheduled full retrain every 6 months regardless of performance signals
5. A major external event occurs (new competitor enters market, pandemic-level disruption)

**Retraining process:**
1. Retrain on all available data up to current month (expanding window)
2. Evaluate new model on the most recent 3-month holdout
3. Compare new model vs current production model on the same holdout
4. If new model RMSE < current model RMSE: deploy to staging, run parallel predictions for 2 weeks, then promote to production
5. Archive old model version — never delete, always maintain rollback capability

---

# Summary

| Section | Key Decisions |
|---|---|
| Problem type | Supervised regression; counterfactual scoring across 5 promotion options |
| Target variable | `items_sold` — directly measures demand without price distortion |
| Modelling strategy | Location-type stratified models or mixed-effects model with store random effects |
| Data grain | One row = store × month × promotion |
| EDA priorities | Promotion × location interaction, seasonality, imbalance check |
| Imbalance handling | Sample weighting or separate baseline + uplift models |
| Evaluation | Walk-forward CV; RMSE, MAE, MAPE, promotion-level MAE |
| Explainability | SHAP values for per-prediction business narratives |
| Deployment | Versioned pipeline, monthly Airflow DAG, PSI-based drift monitoring, triggered retraining |