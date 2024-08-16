
# Predicting Academic Success

This project aims to predict the academic success of students using various machine learning models. The prediction is based on several features, including personal information, academic records, and socio-economic indicators. The project explores different approaches and models to improve the accuracy of predictions.

## Project Overview

The notebook is structured as follows:

1. **Data Loading and Initial Exploration**: 
   - The dataset is loaded and initial exploration is performed to understand the data distribution, missing values, and overall structure.
   - Data includes features like 'Admission grade', 'Age at enrollment', 'Marital status', 'Previous qualification', 'Mother's qualification', 'Father's qualification', 'Unemployment rate', 'GDP', etc.

2. **Data Preprocessing**: 
   - Missing values are handled, categorical variables are encoded, and new interaction features are created.
   - Features such as 'curricular_units_product', 'approved_enrolled_1st_sem_ratio', 'inflation_gdp_ratio', and others are engineered to improve model performance.

3. **Exploratory Data Analysis (EDA)**: 
   - Visualizations are provided to understand the distribution of numerical and categorical features.
   - A correlation matrix is generated to identify relationships between features.

4. **Model Training and Evaluation**:
   - Several machine learning models are employed, including XGBoost, LightGBM, CatBoost, RandomForest, and more.
   - Models are evaluated using accuracy on a validation set. Class weights are calculated to handle class imbalance.
   - Hyperparameter tuning is performed using Optuna to find the best model parameters.

5. **Stacking and Voting Ensemble**: 
   - A stacking ensemble model is created, combining predictions from several base models.
   - A voting classifier is also used to aggregate predictions from multiple models for final predictions.

6. **Feature Importance**:
   - Feature importance is calculated and visualized for the tuned XGBoost model.

7. **Final Predictions**:
   - The final predictions are made on the test dataset, and results are saved to CSV files.

## Requirements

To run the notebook, you'll need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- plotly
- optuna
- xgboost
- lightgbm
- catboost
- scikit-learn

You can install these using pip:

\`\`\`bash
pip install numpy pandas matplotlib seaborn plotly optuna xgboost lightgbm catboost scikit-learn
\`\`\`

## Usage

1. Clone the repository:

\`\`\`bash
git clone <repository-url>
\`\`\`

2. Navigate to the project directory and open the notebook:

\`\`\`bash
cd <project-directory>
jupyter notebook Predicting Academic Success.ipynb
\`\`\`

3. Run the cells in the notebook to execute the code and generate predictions.

## Results

- The notebook generates several CSV files containing predictions made by different models:
  - \`submission_base_xgb.csv\`
  - \`submission_tuned_xgboost.csv\`
  - \`submission.csv\`
  - \`submission_voting_model.csv\`
  
- The accuracy of the models on the validation set is printed, allowing you to compare different approaches.

## Conclusion

This project provides a comprehensive approach to predicting academic success using machine learning techniques. By exploring various models and techniques, including ensemble methods and hyperparameter tuning, the notebook offers a solid foundation for further experimentation and improvement.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
