# Business Case Analysis — Promotion Effectiveness

---

# B1. Problem Formulation

## (a) ML Problem

* **Target:** `items_sold`
* **Features:** store attributes, promotion type, time (month/festival), competition
* **Type:** Supervised Regression

**Why:** Predict continuous sales volume.

---

## (b) Why Items Sold > Revenue

* Revenue is affected by **price/discounts**
* Items sold reflects **true demand**

**Principle:**
Target must align with business objective and avoid distortion

---

## (c) Better Modelling Strategy

Instead of one global model:

* Segment models (urban / rural)
* Cluster-based models
* Hierarchical models

**Why:** Store behavior varies significantly

---

# B2. Data & EDA Strategy

## (a) Data Joining

**Tables:**

* Transactions + Stores + Promotions + Calendar

**Join Keys:**

* `store_id`, `promotion_id`, `transaction_date`

**Final Dataset:**
One row = Store × Month

**Aggregations:**

* Total items sold
* Avg basket size
* Footfall
* Promotion type

---

### Data Pipeline Diagram

```
Transactions ─┐
              ├── Merge ──> Aggregation ──> Final Dataset (Store-Month Level)
Stores ───────┤
Promotions ───┤
Calendar ─────┘
```

---

## (b) EDA Plan

| Analysis               | Purpose             | Impact             |
| ---------------------- | ------------------- | ------------------ |
| Promotion vs Sales     | Best promotion      | Feature importance |
| Store Type vs Sales    | Segment differences | Segmented models   |
| Time Trends            | Seasonality         | Date features      |
| Correlation Heatmap    | Feature relations   | Feature selection  |
| Promotion Distribution | Imbalance check     | Resampling         |

---

## (c) Imbalance Issue

* 80% data = no promotion
* Model bias risk

**Solutions:**

* Resampling
* Weighted training
* Separate models

---

# B3. Evaluation & Deployment

## (a) Train-Test Strategy

* Use **time-based split**
* Train: past data
* Test: recent months

**Why not random?**

* Causes **data leakage**

---

### Time Split Diagram

```
|------ Training ------|--- Testing ---|
Past -------------------------------> Future
```

---

**Metrics:**

* RMSE → penalizes large errors
* MAE → average error

---

## (b) Explaining Model Decisions

Different recommendations = different conditions

**Key drivers:**

* Month / seasonality
* Festival
* Promotion type

**Example:**

* December → Loyalty Points (high demand)
* March → Discount (boost demand)

---

## (c) Deployment Pipeline

### System Flow

```
Raw Monthly Data
       ↓
Preprocessing Pipeline
       ↓
Trained Model (Saved)
       ↓
Predictions (Per Store)
       ↓
Business Recommendations
```

---

### Steps

1. Save model:

```python
joblib.dump(model, 'model.pkl')
```

2. Monthly:

* Load new data
* Apply pipeline
* Predict

3. Automate:

* Cron / Airflow

---

### Monitoring

* RMSE / MAE trends
* Data drift
* Prediction drift

---

### Retraining

* Every 3–6 months OR
* When performance drops

---

# ✅ Summary

* Regression problem predicting `items_sold`
* Use segmented modelling
* Apply time-based validation
* Deploy with monitoring + retraining

---

# End
