# Data Science Portfolio — Clarissa Faith C. Santiago

**BS Data Science | Mapúa University | School of Information Technology**

This repository contains four Jupyter Notebook projects covering machine learning, customer analytics, and time series analysis. Each notebook demonstrates an end-to-end data science workflow — from raw data cleaning and exploratory analysis through to modeling, evaluation, and business or scientific interpretation.

---

## Repository Contents

| Notebook | Type | Key Techniques |
|---|---|---|
| Bank Customer Churn Prediction | Supervised ML | Stacking Ensemble, Random Forest, XGBoost |
| Customer Segmentation with K-Means | Unsupervised ML | K-Means Clustering, EDA, Feature Engineering |
| Houston Temperature Prediction | Time Series Forecasting | ARIMA, Auto-ARIMA, Stationarity Testing |
| Electricity Load Time Series EDA | Time Series EDA | Seasonal Decomposition, ACF/PACF, Trend Analysis |

---

## Project 1: Bank Customer Churn Prediction

**Type:** Group Project  
**File:** `Bank_Customer_Churn_Prediction.ipynb`

### Overview
This project builds a stacking ensemble model to predict whether a bank customer is likely to churn (leave the bank). Rather than relying on a single algorithm, the model combines the strengths of two base learners — Random Forest and XGBoost — and feeds their probability outputs into a Logistic Regression meta-model for the final prediction.

### Dataset
- **Source:** [Churn Modelling Classification Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 bank customer records
- **Target variable:** `Exited` (1 = churned, 0 = retained)

### Methodology

**Data Preprocessing**
- Dropped non-informative columns (`RowNumber`, `Surname`, `CustomerId`) from the model input; `CustomerId` and `Surname` are preserved separately in a tracker dataframe for result display
- Applied **Label Encoding** to `Gender` (binary feature)
- Applied **One-Hot Encoding** to `Geography` (multi-class: France, Germany, Spain) with `drop_first=True` to avoid multicollinearity

**Feature Engineering**
- Created `AgeGroup` — customers binned into five age categories (Young Adults, Early Middle-Age, Mid-Life Adults, Pre-Retirement, Retirees & Seniors) to help the model generalize across age-related behavioral patterns
- Created `BalanceSalaryRatio` — balance normalized by estimated salary to capture relative financial stability, preventing high-salary customers from distorting the balance feature's importance; values are capped at the 99th percentile to prevent extreme outliers from distorting model predictions

**Modeling: Stacking Ensemble**

The stacking pipeline follows these steps:

1. **Base Model 1 — Random Forest** (`n_estimators=100`, `max_depth=10`, `class_weight="balanced"`) accounts for class imbalance using balanced weights
2. **Base Model 2 — XGBoost** (`n_estimators=100`, `max_depth=5`, `eval_metric="logloss"`) uses `scale_pos_weight` computed from the actual class ratio in the training set to handle imbalance
3. **Out-of-Fold (OOF) Predictions** are generated for both base models using 5-fold cross-validation. Each fold is predicted by a model that was never trained on it, producing honest, unbiased probability estimates and preventing data leakage into the meta-model
4. **Meta-Feature Scaling** is applied via `StandardScaler` before training the meta-model, ensuring Logistic Regression receives consistently scaled inputs
5. **Meta-Model — Logistic Regression** is trained on the scaled OOF predictions, learning the optimal blend of both base models' outputs into a final churn probability

**Evaluation**
- Cross-validation accuracy: **85.67%**
- Test set accuracy: **86.40%**
- The OOF-based cross-validation ensures the CV accuracy reflects honest generalization performance. The moderate gap between CV and test accuracy indicates reasonable generalization without severe overfitting. The model performs well at identifying retained customers, but is less confident in flagging churners — effective in most cases but may occasionally miss customers who are likely to leave.

**Feature Importance**
- Random Forest top features: `Age`, `NumOfProducts`, `AgeGroup`
- XGBoost top features: `NumOfProducts`, `IsActiveMember`, `Age`
- Averaged importance (rank averaging): `NumOfProducts` ranked as the most influential predictor overall, followed by `Age` and `IsActiveMember`

### Key Findings
Number of products held and customer age are the strongest predictors of churn. Active membership status also significantly impacts retention. Features like `HasCrCard`, `Tenure`, and `Gender` contribute minimally to prediction.

### Libraries
`pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `xgboost`

---

## Project 2: Customer Segmentation with K-Means Algorithm

**Type:** Group Project  
**File:** `Customer_Segmentation_with_KMeans_Algorithm.ipynb`

### Overview
This project applies K-Means clustering to over 500,000 retail transaction records from a UK-based online gift store to segment customers into distinct behavioral groups. The goal is to help businesses understand different customer types and develop targeted marketing strategies for each segment.

### Dataset
- **Source:** [Online Retail Customer Clustering Dataset](https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering/data)
- **Coverage:** December 2010 – December 2011, UK-based online gift store
- **Size:** 500,000+ transaction records

### Methodology

**Data Preprocessing**
- Removed rows with missing `CustomerID` values
- Filtered out invalid transactions (negative quantities, zero unit prices)
- Retained apparent duplicate rows as they represent distinct items within the same transaction, not true duplicates
- Computed `TotalPrice` per line item (`Quantity × UnitPrice`)

**Exploratory Data Analysis**
- Unit price distribution showed most products priced under £10
- Top 10 most purchased products identified by total quantity sold
- Revenue analysis by country confirmed that the UK dominates sales, supporting a UK-focused segmentation approach

**Feature Engineering**
- Aggregated transaction-level data into customer-level metrics:
  - `TransactionCount` — total number of unique invoices (purchase frequency)
  - `Quantity` — total items purchased
  - `TotalPrice` — total revenue generated per customer
- Applied StandardScaler to normalize features before clustering, ensuring no single feature dominates due to scale differences

**Clustering: K-Means**
- Applied the Elbow Method (WCSS) to determine the optimal number of clusters
- While the elbow suggested k=2, k=4 was selected to capture more nuanced behavioral differences relevant to practical marketing use cases

**Resulting Customer Segments**

| Segment | Behavior Profile |
|---|---|
| Occasional Buyers | Low frequency, low spend |
| Regular Buyers | Moderate frequency and spend |
| Loyal Customers | High frequency, consistent spend |
| High Bulk Buyers | High quantity purchases, significant revenue contribution |

### Business Recommendations
- **High Bulk Buyers & Loyal Customers:** Retention strategies, exclusive rewards, early access programs
- **Regular Buyers:** Upsell incentives, product bundling offers
- **Occasional Buyers:** Re-engagement campaigns, discount promotions to increase purchase frequency

### Libraries
`pandas` `numpy` `matplotlib` `seaborn` `scikit-learn`

---

## Project 3: Houston Temperature Prediction — Time Series Analysis

**Type:** Solo Project  
**File:** `Temperature_Prediction_Time_Series_Analysis.ipynb`

### Overview
This project forecasts Houston, Texas's monthly average temperature for the next 6 months beyond the available data (December 2017 – May 2018) using ARIMA-based time series models. Two approaches are developed and compared — manually specified ARIMA and automatically optimized Auto-ARIMA — first validated on a held-out 12-month test set against real values, then retrained on the full dataset to generate the final future forecast.

### Dataset
- **Source:** [Hourly Weather Dataset](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data?select=temperature.csv)
- **Coverage:** 5 years of hourly temperature readings from 30 US/Canada cities and 6 Israeli cities
- **Scope:** Houston, Texas only
- **Resampled** to monthly averages for modeling

### Methodology

**Data Preprocessing & Feature Engineering**
- Parsed datetime index and extracted Houston temperature column
- Handled missing values via linear interpolation with forward/backward fill for edge cases
- Resampled hourly data to monthly averages

**Exploratory Data Analysis**
- Monthly temperature plot revealed clear seasonality with no significant long-term trend
- July recorded the highest average monthly temperature (301.98 K)
- January recorded the lowest average monthly temperature (284.92 K)

**Stationarity Testing: ADF Test**
- ADF Statistic ≈ -0.045 (greater than 5% critical value of -2.89)
- p-value ≈ 0.95 (greater than 0.05)
- Result: Series is non-stationary → first differencing required
- Note: Differencing is handled internally by the models (d=1), not applied as a preprocessing step, to ensure a fair comparison

**ACF and PACF Analysis**
- ACF showed significant autocorrelation up to lag 3 → q = 3
- PACF showed cutoff at lag 2 → p = 2
- These findings inform the Manual ARIMA parameter selection and confirm d = 1 is appropriate

**Train/Test Split**
- Training set: November 2012 – December 2016
- Test set: January 2017 – November 2017 (12 months, held out)
- Both models trained on training set and evaluated against real held-out values for valid out-of-sample assessment

**Models Compared**

*Auto-ARIMA — order (3, 1, 2), seasonal (0, 0, 0)[12]*
- Automatically selected optimal parameters via AIC-based grid search over the full parameter space (`stepwise=False`)
- `d=1` fixed based on ADF test result confirming non-stationarity
- `D=0` set to prevent seasonal over-differencing, justified by the stable seasonal amplitude observed in EDA
- The selected seasonal order (0, 0, 0)[12] indicates that no seasonal AR, MA, or differencing terms were needed — the regular ARIMA(3, 1, 2) terms were sufficient to capture the seasonal pattern
- **Test Set Metrics — MAE: 0.919 K | MSE: 1.757 | RMSE: 1.325 K**

*Manual ARIMA — order (2, 1, 3)*
- Parameters derived from ACF/PACF analysis; d=1 applied internally
- **Test Set Metrics — MAE: 0.972 K | MSE: 1.896 | RMSE: 1.377 K**

**Future Forecast (Dec 2017 – May 2018)**
- Both models retrained on full dataset after validation
- Forecasts from both models are nearly identical (max difference: 0.15 K)
- Predicted trajectory: winter low ~290 K (January 2018) rising to ~298.5 K by May 2018, consistent with Houston's historical seasonal pattern

### Key Findings
Both models performed comparably well on the held-out test set, with average errors below 1 Kelvin. Auto-ARIMA achieved marginally lower error metrics across all measures and is selected as the primary model. A notable finding is that Auto-ARIMA required domain-informed constraints — derived from ADF testing and EDA — to avoid over-differencing and produce a meaningful forecast. The final 6-month forecasts from both models are in strong agreement, supporting confidence in the predicted temperature trajectory.

### Libraries
`pandas` `numpy` `matplotlib` `seaborn` `statsmodels` `pmdarima` `scikit-learn`

---

## Project 4: Time Series EDA on Electricity Load (PJME)

**Type:** Solo Project  
**File:** `Time_Series_Exploratory_Data_Analysis_on_Electricity_Load.ipynb`

### Overview
This notebook performs an in-depth exploratory data analysis on the PJM East (PJME) hourly electricity load dataset, analyzing consumption patterns across hourly, daily, monthly, and yearly time scales. The project uses seasonal decomposition and ACF/PACF analysis to uncover the structural components of electricity demand.

### Dataset
- **Source:** `PJME_hourly.csv` — PJM Interconnection hourly electricity load data
- **Coverage:** 2002–2018
- **Variable:** `PJME_MW` — electricity load in megawatts

### Methodology

**Data Exploration**
- Parsed and indexed datetime column
- Resampled data to yearly, monthly, and daily averages for multi-scale trend analysis

**Consumption Pattern Analysis**
- Extracted `Hour`, `DayOfWeek`, and `Month` features from the datetime index
- Computed average electricity load by each time granularity

**Key Findings by Time Dimension**

| Dimension | Peak | Lowest |
|---|---|---|
| Hour of day | 7 PM (evening residential/commercial activity) | 4 AM (overnight inactivity) |
| Day of week | Tuesday (peak business operations) | Sunday (reduced industrial and commercial activity) |
| Month | July (summer cooling demand) | April (mild weather, low HVAC demand) |

**Trend Analysis**
- Overall yearly average load declined from 2002 to 2018, with peaks around 2006–2008
- Strong repeating seasonal cycles visible in monthly averages
- Daily load shows frequent fluctuation driven by usage patterns

**Time Series Decomposition**
- Applied seasonal decomposition to separate trend, seasonality, and residual components from the raw load signal
- ACF and PACF plots used to identify autocorrelation structure in the series

### Libraries
`pandas` `matplotlib` `statsmodels`

---

## Technical Stack

| Category | Tools & Libraries |
|---|---|
| Language | Python 3 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost |
| Time Series | statsmodels, pmdarima |
| Environment | Jupyter Notebook |

---

## Author

**Clarissa Faith Cordero Santiago**
BS Data Science — Mapúa University, School of Information Technology
Manila, Philippines

- **Portfolio:**datascienceportfol.io/clarissafaith
- **Email:** cfcsantiago@mymail.mapua.edu.ph / santiagoclarissafaith@gmail.com
- **LinkedIn:** [linkedin.com/in/clarissa-faith-santiago](https://www.linkedin.com/in/clarissa-faith-santiago)
