MLCEFE – Multi-Level Credit Evaluation Framework with Weighted Ensemble Learning and Explainable Insights

**Overview**This project presents an optimized credit risk prediction pipeline that uses multiple machine learning models in a weighted ensemble framework. The main application is in loan approval classification, but the pipeline can be adapted to other binary classification problems, especially in financial risk analysis.

The pipeline focuses on improving predictive performance, handling class imbalance, and providing interpretability through feature importance and visualization techniques. It integrates advanced preprocessing, dimensionality reduction, and ensemble learning to achieve strong performance.

**Key Features**

*   Weighted Ensemble of XGBoost, LightGBM, Gradient Boosting, and Balanced Random Forest Classifier.
    
*   Class imbalance handling using SMOTE (Synthetic Minority Oversampling Technique).
    
*   Dimensionality reduction with PCA (Principal Component Analysis).
    
*   End-to-end pipeline from data loading to evaluation.
    
*   Multiple performance evaluation metrics including ROC-AUC, F1-Score, and confusion matrix.
    
*   Feature importance visualizations for interpretability.
    

**Technology Stack**

*   Python 3.8+
    
*   Pandas, NumPy for data manipulation
    
*   Matplotlib, Seaborn for data visualization
    
*   Scikit-learn for preprocessing, modeling, and evaluation
    
*   XGBoost and LightGBM for gradient boosting algorithms
    
*   Imbalanced-learn for SMOTE and Balanced Random Forest
    
*   TensorFlow/Keras for optional autoencoder-based feature extraction
    

**Project Structure**

*   Final.ipynb: Jupyter Notebook containing the complete implementation.
    
*   requirements.txt: List of dependencies to run the project.
    
*   data/loan\_data.csv: Input dataset (must be provided by the user).
    
*   results/: Directory to store generated plots such as confusion matrix, ROC curve, and feature importance.
    

**Dataset**The dataset should contain loan application records with both categorical and numerical features, along with a binary target variable indicating loan approval status.

Typical features include:

*   Categorical: Gender, Marital Status, Employment Type, Education, Property Area.
    
*   Numerical: Applicant Income, Co-applicant Income, Loan Amount, Credit Score.
    
*   Target Variable: Loan\_Status (Y for approved, N for rejected).
    

A publicly available source for such data is the Loan Approval dataset on Kaggle.

**Workflow**

Step 1: Data Loading

*   Load dataset from CSV.
    
*   Handle missing values and type conversions.
    

Step 2: Preprocessing

*   Encode categorical variables using label encoding.
    
*   Standardize numerical features for better model performance.
    
*   Optionally apply PCA for dimensionality reduction.
    

Step 3: Handling Class Imbalance

*   Use SMOTE to oversample the minority class.
    
*   Achieve a balanced dataset to reduce bias in predictions.
    

Step 4: Model BuildingModels used include:

*   Gradient Boosting Classifier
    
*   XGBoost Classifier
    
*   LightGBM Classifier
    
*   Balanced Random Forest Classifier
    

Predictions from these models are combined using a weighted soft-voting ensemble.

Step 5: Evaluation

*   Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
    
*   Visualize performance with ROC curves and confusion matrices.
    
*   Display feature importance for explainability.
    

**Example Results**

Model Performance Table:Gradient Boosting – Accuracy: 0.84, ROC-AUC: 0.88XGBoost – Accuracy: 0.86, ROC-AUC: 0.90LightGBM – Accuracy: 0.85, ROC-AUC: 0.89Balanced Random Forest – Accuracy: 0.83, ROC-AUC: 0.87Weighted Ensemble – Accuracy: 0.88, ROC-AUC: 0.92

**Installation and Usage**

1.  Clone the repository:git clone [https://github.com/yourusername/MLCEFE.git](https://github.com/yourusername/MLCEFE.git)cd MLCEFE
    
2.  Install dependencies:pip install -r requirements.txt
    
3.  Place your dataset:Save loan\_data.csv in the data folder.
    
4.  Run the notebook:jupyter notebook Final.ipynb
    

**Future Enhancements**

*   Deploy as an interactive web application using Streamlit.
    
*   Integrate SHAP for advanced interpretability.
    
*   Add hyperparameter tuning with Optuna.
    
*   Extend to multi-class credit risk grading.
    

**Acknowledgements**

*   Scikit-learn for core machine learning utilities.
    
*   Imbalanced-learn for SMOTE and ensemble methods.
    
*   LightGBM and XGBoost for gradient boosting algorithms.
    
*   Kaggle for publicly available datasets.
    

**License**This project is licensed under the MIT License, allowing free use, modification, and distribution.
