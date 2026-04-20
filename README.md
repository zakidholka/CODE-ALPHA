# CodeAlpha Data Science Internship

> **Intern:** [Your Name]
> **Domain:** Data Science
> **Company:** [CodeAlpha](https://www.codealpha.tech)

---

## Task Overview

| Task | Title | Status |
|------|-------|--------|
| Task 1 | Iris Flower Classification | Completed |
| Task 2 | Unemployment Analysis with Python | Completed |
| Task 3 | Car Price Prediction with Machine Learning | Completed |
| Task 4 | Sales Prediction using Python | Completed |

---

## Task 1 - Iris Flower Classification

### Objective
Train a machine learning model to classify Iris flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.

### Dataset
- **Source:** Kaggle - iriscsv
- **Records:** 150 samples | **Features:** 4 | **Classes:** 3

### Google Colab Setup
```python
# Step 1 - Install kagglehub
!pip install kagglehub -q

# Step 2 - Download dataset
import kagglehub
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)
```

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
```

### Models Trained

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~93% |
| Decision Tree | ~93% |
| Random Forest | ~93% |
| SVM (RBF) | ~97% - Best |
| K-Nearest Neighbors | ~93% |

### Key Results
- **Best Model:** SVM (RBF Kernel) with 96.7% accuracy
- **Top Features:** Petal length and Petal width
- **All 3 species** classified with high precision and recall

### Plots Saved
1. fig1_distributions.png - Feature histograms by species
2. fig2_pairplot.png - Pairwise relationships
3. fig3_correlation.png - Correlation heatmap
4. fig4_model_results.png - Model comparison + confusion matrix
5. fig5_feature_importance.png - Random Forest importance

---

## Task 2 - Unemployment Analysis with Python

### Objective
Analyze India's unemployment data, investigate the impact of COVID-19 lockdowns, identify regional and seasonal patterns, and present actionable insights.

### Dataset
- **Source:** Kaggle - unemployment-in-india by gokulrajkmv
- **Records:** 290 rows | **Period:** Jan 2020 to Oct 2020 | **States:** 29

### Google Colab Setup
```python
# Step 1 - Install kagglehub
!pip install kagglehub -q

# Step 2 - Download dataset
import kagglehub
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Path to dataset files:", path)
```

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
```

### Key Findings

| Period | Avg Unemployment Rate |
|--------|-----------------------|
| Pre-COVID (Jan-Feb 2020) | 6.42% |
| Lockdown (Mar-May 2020) | 16.84% - spike of +10.4 pp |
| Post-Lockdown (Jun-Oct 2020) | 8.08% |

- **Peak Rate:** 47.12% in Jharkhand (April 2020)
- **Most Impacted State:** Jharkhand (+24.33 pp during lockdown)
- **Least Impacted State:** Uttar Pradesh (+2.83 pp)
- **Correlation (Unemployment vs Labour Participation Rate):** -0.62

### Plots Saved
1. fig1_national_trend.png - National unemployment + LPR trend
2. fig2_period_comparison.png - Pre vs Lockdown vs Post comparison
3. fig3_regional_analysis.png - 6-region breakdown
4. fig4_state_heatmap.png - All states across all months
5. fig5_state_impact.png - Most vs least affected states
6. fig6_correlation.png - Scatter and correlation matrix
7. fig7_monthly_distribution.png - Monthly boxplots

---

## Task 3 - Car Price Prediction with Machine Learning

### Objective
Predict the selling price of used cars based on features like brand, present price, kilometers driven, fuel type, transmission, and ownership history.

### Dataset
- **Source:** Kaggle - car-price-predictionused-cars by vijayaadithyanvg
- **Columns:** Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
- **Target:** Selling_Price (in Lakhs)

### Google Colab Setup
```python
# Step 1 - Install kagglehub
!pip install kagglehub -q

# Step 2 - Download dataset
import kagglehub
path = kagglehub.dataset_download("vijayaadithyanvg/car-price-predictionused-cars")
print("Path to dataset files:", path)
```

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
```

### Feature Engineering

| New Feature | Description |
|-------------|-------------|
| Car_Age | 2024 minus Year of manufacture |
| Brand | Extracted from Car_Name column |
| Brand_Tier | Luxury / Mid / Budget classification |

### Model Results

| Model | R2 Score |
|-------|----------|
| Linear Regression | ~0.85 |
| Ridge Regression | ~0.87 |
| Lasso Regression | ~0.86 |
| Decision Tree | ~0.82 |
| Random Forest | ~0.93 |
| Gradient Boosting | ~0.93 - Best |

### Key Findings
- Present_Price is the strongest predictor of selling price
- Diesel cars fetch higher resale prices than Petrol
- Dealer-sold cars are priced higher than individual sellers
- Automatic transmission adds a price premium over Manual
- More previous owners leads to lower selling price

### Plots Saved
1. fig1_price_distribution.png - Price distribution and brand analysis
2. fig2_feature_vs_price.png - Numeric features vs price
3. fig3_categorical_vs_price.png - Fuel, seller, transmission boxplots
4. fig4_violin_plots.png - Violin plots by fuel type and transmission
5. fig5_correlation_heatmap.png - Feature correlation heatmap
6. fig6_model_comparison.png - All 6 models compared
7. fig7_actual_vs_predicted.png - Predictions vs actual + residuals
8. fig8_feature_importance.png - Feature importance bar and pie

---

## Task 4 - Sales Prediction using Python

### Objective
Predict future sales based on advertising spend across TV, Radio, and Newspaper channels. Analyze how advertising changes impact sales and provide actionable marketing budget recommendations.

### Dataset
- **Source:** Kaggle - advertisingcsv by bumba5341
- **Records:** 200 | **Features:** TV, Radio, Newspaper | **Target:** Sales

### Google Colab Setup
```python
# Step 1 - Install kagglehub
!pip install kagglehub -q

# Step 2 - Download dataset
import kagglehub
path = kagglehub.dataset_download("bumba5341/advertisingcsv")
print("Path to dataset files:", path)
```

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
```

### Feature Engineering

| New Feature | Description |
|-------------|-------------|
| Total_Spend | TV + Radio + Newspaper combined |
| TV_Radio | TV x Radio interaction (synergy effect) |
| TV_News | TV x Newspaper interaction |
| Radio_News | Radio x Newspaper interaction |
| log_TV | Log transform of TV spend |
| log_Radio | Log transform of Radio spend |
| TV_share | TV as percentage of total budget |

### Channel Analysis

| Channel | Correlation with Sales | Impact |
|---------|----------------------|--------|
| TV | 0.78 - Highest | Strong positive |
| Radio | 0.58 | Moderate positive |
| Newspaper | 0.23 - Lowest | Weak positive |

### Model Results

| Model | R2 Score |
|-------|----------|
| Linear Regression | ~0.87 |
| Ridge Regression | ~0.87 - Best |
| Lasso Regression | ~0.87 |
| Decision Tree | ~0.82 |
| Random Forest | ~0.83 |
| Gradient Boosting | ~0.87 |

### Business Recommendations
1. TV has the highest correlation with sales - prioritize it
2. Radio delivers strong ROI at lower cost than TV
3. Newspaper shows lowest impact - consider reallocating that budget
4. Running TV and Radio campaigns simultaneously creates a synergy boost
5. High-budget multi-channel approach gives maximum predicted sales

### Plots Saved
1. fig1_overview.png - Sales distribution and budget share
2. fig2_ad_impact.png - Channel spend vs sales scatter plots
3. fig3_correlation.png - Heatmap and ranked correlation
4. fig4_pairplot.png - All features pairplot
5. fig5_model_comparison.png - 6 models compared
6. fig6_actual_vs_predicted.png - Predictions, residuals, histogram
7. fig7_feature_importance.png - Importance bar and channel pie
8. fig8_whatif_forecast.png - Budget scenario comparison
9. fig9_budget_simulation.png - Response curves and allocation

---

## General Colab Setup (All Tasks)

### Install all libraries at once
```python
!pip install numpy pandas matplotlib seaborn scikit-learn kagglehub -q
```

### Download and display plots inline
```python
# Plots are shown inline with plt.show() after each figure
# They are also saved automatically to SAVE_DIR

import os
SAVE_DIR = "/content/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save example
plt.savefig(f"{SAVE_DIR}/fig1_example.png", bbox_inches="tight", dpi=130)
plt.show()
```

### Download all plots as ZIP
```python
import zipfile, os
from google.colab import files

SAVE_DIR = "/content/plots"
zip_path = "/content/all_plots.zip"

with zipfile.ZipFile(zip_path, "w") as zf:
    for fname in sorted(os.listdir(SAVE_DIR)):
        if fname.endswith(".png"):
            zf.write(os.path.join(SAVE_DIR, fname), fname)
            print(f"Added: {fname}")

files.download(zip_path)
print("Download started!")
```

---

## Contact

- **Website:** www.codealpha.tech
- **Email:** services@codealpha.tech
- **WhatsApp:** +91 9336576683

---

*CodeAlpha Data Science Internship - All 4 tasks completed successfully.*
