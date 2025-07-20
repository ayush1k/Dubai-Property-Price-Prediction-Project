# Dubai Property Price Prediction and Key Drivers Analysis

## Project Overview

This repository contains a data science project focused on predicting optimal property pricing for new developments in Dubai and identifying the key factors that drive property value. Developed as an assignment for ProjektAnalytics, a decision sciences company, the project emphasizes actionable insights and model explainability for non-technical stakeholders.

The solution leverages advanced machine learning techniques, including **XGBoost with GPU acceleration** and **Target Encoding** for high-cardinality categorical features, to build a robust and efficient predictive model.

## Business Problem

The core business problem addressed is:
**"How can we accurately predict optimal property prices for new developments in Dubai, and what are the most influential factors driving these values, enabling businesses (like property developers and investors) to make informed decisions and scale rapidly?"**

Accurate property valuation is crucial for strategic planning, competitive pricing, and maximizing return on investment in the dynamic real estate sector.

## Key Features & Methodologies

* **Data Loading & Preprocessing:** Efficient handling of a large real estate transaction dataset, including cleaning missing values, converting data types, and handling outliers.
* **Feature Engineering:** Creation of relevant features and careful selection of influential variables for the model.
* **Target Variable Transformation:** Application of logarithmic transformation (`np.log1p`) to the target variable (`actual_worth`) to handle skewness and improve model performance.
* **High-Cardinality Categorical Encoding (Target Encoding):** Utilized `category_encoders.TargetEncoder` to effectively encode high-cardinality features like `building_name` and `area_name`, preventing memory issues associated with One-Hot Encoding and capturing their inherent value.
* **Predictive Modeling (XGBoost Regressor):** Implementation of `xgboost.XGBRegressor`, a powerful gradient boosting algorithm known for its high accuracy on tabular data.
* **GPU Acceleration:** Configured XGBoost to use `tree_method='gpu_hist'`, significantly speeding up model training on large datasets in GPU-enabled environments (e.g., Google Colab T4 GPU).
* **Model Evaluation:** Assessment of model performance using key regression metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2).
* **Model Explainability (Feature Importance):** Extraction and visualization of feature importance scores from the XGBoost model to clearly articulate which factors contribute most to property price predictions.
* **What-If Scenarios / Counterfactuals:** Demonstration of how changes in key property attributes (e.g., building name, area size) impact predicted prices, providing tangible insights for strategic decision-making.
* **Conceptual Causal Inference:** Introduction of a Directed Acyclic Graph (DAG) framework to hypothesize causal relationships, identifying root causes, counterfactuals, and interventions to move beyond correlation and provide actionable business recommendations.

## Data Source

The primary dataset used is:
* **Dubai Real Estate Transactions Dataset**
    * Source: [Kaggle](https://www.kaggle.com/datasets/alexefimik/dubai-real-estate-transactions-dataset)
    * File used: `Transactions.csv`

## Results & Key Findings

The GPU-optimized XGBoost model achieved the following performance on the test set:

* **Mean Absolute Error (MAE):** `1,017,788.05 AED`
* **Root Mean Squared Error (RMSE):** `3,467,682.21 AED`
* **R-squared (R2):** `0.6582`

**Key Findings (Feature Importance):**

The model identified the following as the most influential factors in predicting property prices:

1.  **`area_sqft` (Property Size):** The most significant driver, indicating that larger properties command higher values.
2.  **`building_name` (Building Quality/Brand):** A crucial determinant, reflecting the impact of specific building reputation, quality, and amenities.
3.  **`property_usage`:** Whether the property is residential, commercial, or mixed-use.
4.  **`area_name` (Location):** The specific geographical area of the property.
5.  **`property_type`:** The broad category of the property (e.g., Apartment, Villa).

## Actionable Insights

1.  **Strategic Building and Location Selection:** Focus on high-value, established communities and reputable building brands for new developments and investments.
2.  **Optimize Property Design for Value:** Prioritize spacious layouts, efficient floor plans, and optimal room configurations that maximize usable area, as these directly correlate with higher predicted prices.
3.  **Data-Driven Pricing Strategy:** Utilize the predictive model as a dynamic pricing tool for new launches, inputting property specifications to determine optimal asking prices.
4.  **Informed Investment Decisions for Existing Properties:** Leverage the model to evaluate potential returns and guide renovation strategies that focus on high-impact features to maximize resale value.

## How to Run the Code

1.  **Clone the Repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```

2.  **Google Colab (Recommended):**
    * Open the `GPU_Optimized_Dubai_Property_Price_Prediction_and_Key_Drivers_Analysis.ipynb` notebook in Google Colab.
    * **Enable GPU Runtime:** Go to `Runtime > Change runtime type` and select `T4 GPU` as the hardware accelerator.
    * **Install Dependencies:** Run the following commands in a Colab cell:
        ```python
        !pip install xgboost
        !pip install category_encoders
        ```
    * Run all cells in the notebook (`Runtime > Run all`).

3.  **Local Environment:**
    * **Install Dependencies:**
        ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn xgboost category_encoders kagglehub
        ```
    * **Kaggle API Key (for `kagglehub`):** If running locally, you'll need to set up your Kaggle API key.
        * Go to Kaggle, click on your profile picture, then "My Account".
        * Under the "API" section, click "Create New API Token" to download `kaggle.json`.
        * Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<Windows-username>\.kaggle\` (Windows).
    * Run the Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
        Then open the `.ipynb` file.

## Project Structure


.
├── GPU_Optimized_Dubai_Property_Price_Prediction_and_Key_Drivers_Analysis.ipynb  # Main Jupyter Notebook
├── README.md                                                                    # This file
└── images/                                                                      # Directory for plots (e.g., eda_plots.png, feature_importance_plot.png)

*(Note: You will need to manually save your generated plots into the `images/` directory after running the notebook in Colab or locally)*

## Future Improvements

* **Advanced Feature Engineering:** Explore creating more sophisticated features (e.g., distance to nearest metro/mall if coordinates become available, or a "luxury score" for buildings).
* **Hyperparameter Optimization:** Conduct more exhaustive hyperparameter tuning for the XGBoost model using techniques like Grid Search or Randomized Search.
* **Time Series Analysis:** Incorporate the `transaction_date` more deeply to model market trends, seasonality, and predict future price movements.
* **External Data Integration:** Explore the integration of macroeconomic indicators or specific development project data to further enhance predictive power and causal understanding.
* **SHAP/LIME Implementation:** For individual property pricing decisions, implement SHAP or LIME to provide highly localized and interpretable explanations for each prediction.

## Contact

For any questions or further discussion, please feel free to reach out:

[Your Full Name]
[Your Email Address]
[Your LinkedIn Profile URL (Optional)]

---
