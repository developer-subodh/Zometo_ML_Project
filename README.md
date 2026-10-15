# 🚀 Enhanced Zomato Data Science Project: End-to-End Predictive Analytics

An advanced data science and machine learning repository focused on optimizing restaurant pricing and discovering menu success patterns on Zomato's platform. This project handles a cleaned, enhanced dataset of **123,657 rows** and implements both **Regression** (Price Prediction) and **Imbalanced Classification** (Bestseller Discovery) models.

---

## 📌 Project Overview & Goals
The objective of this project was to transition from standard exploratory analysis into rigorous machine learning development. The repository is split into two primary business problems:
1. **Regression Engine:** Predict the precise market price (`Prices`) or log-transformed price (`Log_Price`) of a menu item based on restaurant popularity, city metrics, and cuisine types.
2. **Classification Engine:** Identify whether a newly listed menu item will become a `Bestseller` or `Must Try` category winner (`Is_Bestseller = 1`).

---

## 🔬 Key Learnings & Technical Breakdown

### 1. Advanced Exploratory Data Analysis (EDA)
* **The Price Skewness Trap:** Initial visualization showed that **90% of menu items are budget-friendly (< ₹500)**. However, high-end luxury outliers stretched up to ₹12,000. Training models on raw prices heavily biased the error loss functions.
* **Rating & Price Independence:** Discovered a weak linear correlation between prices and ratings. Premium items (`Is_Expensive = 1`) scored an average rating of **3.93**, while budget items scored **3.88**—proving that price does not dictate user satisfaction.
* **The "L-Shape" Funnel Pattern:** Visualizing `Prices vs Total_Votes` revealed distinct heteroscedasticity. Highly popular restaurants (>800 votes) maintain very tight, stable pricing structures capping items under ₹500.

### 2. Feature Engineering Master Sequence (No Data Leakage)
To completely prevent the risk of **Data Leakage**, a strict operational pipeline was strictly enforced:
* **Train-Test Split First:** The dataset was partitioned into an 80% training set and a 20% validation set before applying any processing parameters.
* **Categorical Encoding Strategy:** 
  * `Cuisine` (High Cardinality) was converted via **Label Encoding**.
  * `City` was processed via **One-Hot Encoding** with `drop_first=True` to bypass the Dummy Variable Trap.
* **Aggregated Target Features (The Secret Sauce):** Engineered historical contextual features including `Avg_Price_Restaurant`, `Avg_Price_Cuisine`, and `Avg_Price_City`. These features served as mathematical anchors, drastically dropping prediction variance.
* **Selective Scaling:** Applied `StandardScaler` exclusively to raw numerical streams and label-encoded vectors. Binary dummy arrays (0/1) and the target values were deliberately bypassed to safeguard structural interpretation.

### 3. Regression Modeling & Evaluation
We compared linear models against non-linear boosting ensembles to capture structural complexities:

| Algorithm | R2 Score (Accuracy) | Mean Absolute Error (MAE) | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.4439 | 0.3673 | Baseline (Failed to capture non-linearities) |
| **Random Forest Regressor** | 0.9831 | 0.0407 | Exceptional Performance |
| **XGBoost Regressor** | **0.9949** | **0.0221** | 🏆 **Ultimate Champion Model** |

* **Mathematical Inversion:** Predictions were successfully converted back from log-space to Indian Rupees (INR) using `np.exp()` for clean commercial verification.
* **Overfitting Rigorous Audit:** XGBoost achieved a Train R2 of **0.9957** and a Test R2 of **0.9949**. The negligible delta of **0.0008** systematically proved that the model learned robust patterns instead of memorizing data.

### 4. Classification & Handling Data Imbalance
Transitioning to predict `Is_Bestseller`, we confronted a severe data imbalance where ~83% of the data belonged to the majority class (Normal items).

* **The Accuracy Paradox:** The initial un-tuned classifiers boasted a misleading **89% overall accuracy**, but revealed a critically low **52% Recall** for bestselling items. The model was consistently missing half of the commercial successes.
* **The Precision-Recall Trade-Off Solution:** Implemented hyperparameter tuning utilizing XGBoost's in-built minority class optimization parameter (`scale_pos_weight`). 
* **The Sweet Spot Config:** Lowering the penalty ratio to **2.5** delivered the final production-ready matrix:

```text
--- 🏆 Final Optimized Balanced XGBoost Classifier ---
              precision    recall  f1-score   support
           0       0.94      0.90      0.92     20503
           1       0.59      0.71      0.65      4229

    Accuracy: 87.00% | Macro Avg Recall: 81.00%
```
* **Business Significance:** The final architecture successfully caught **71% of all true bestsellers** while maintaining a solid **59% Precision**, dramatically reducing false marketing alarms for restaurant management.

---

## 🛠️ Tech Stack & Libraries Used
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost

---

## 📈 How to Run
1. Clone this repository.
2. Ensure you have the `zomato_enhanced.csv` file inside the root directory.
3. Open the Jupyter Notebook or execute the Python workflow scripts sequentially:
   ```bash
   python features_pipeline.py
   python train_models.py
   ```
