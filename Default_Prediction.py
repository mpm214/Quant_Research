# Import necessary libraries for model testing and handling class imbalance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Quant_Research/Task 3 and 4_Loan_Data.csv'
loan_data = pd.read_csv(file_path)

loan_data.drop(columns=['customer_id'], inplace=True)

original_loan_amounts = loan_data['loan_amt_outstanding']

X = loan_data.drop(columns=['default'])
y = loan_data['default']

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X, y, test_size=0.3, random_state=214)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=214)
X_train_res, y_train_res = sm.fit_resample(X_train_original, y_train_original)

X_train_df = pd.DataFrame(X_train_res, columns=loan_data.drop(columns=['default']).columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_original)

logistic_regression_model = LogisticRegression(random_state=214, class_weight='balanced')
decision_tree_model = DecisionTreeClassifier(random_state=214)
random_forest_model = RandomForestClassifier(random_state=214)

param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search_rf = GridSearchCV(random_forest_model, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_rf.fit(X_train_res, y_train_res)
random_forest_model = grid_search_rf.best_estimator_

logistic_regression_model.fit(X_train_scaled, y_train_res)
decision_tree_model.fit(X_train_res, y_train_res)
random_forest_model.fit(X_train_res, y_train_res)

# Function to calculate expected loss
def calculate_expected_loss(loan_amount, Default_Prob, recovery_rate=0.10):
    return loan_amount * Default_Prob * (1 - recovery_rate)

# Function to predict default probability and expected loss for a given model
def predict_expected_loss(model, X_test, loan_amounts, use_scaler=False, recovery_rate=0.10):
    if use_scaler:
        Default_Prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        Default_Prob = model.predict_proba(X_test)[:, 1]

    expected_loss = [calculate_expected_loss(loan_amt, p, recovery_rate) for loan_amt, p in zip(loan_amounts, Default_Prob)]
    return Default_Prob, expected_loss

# Evaluate all models and store Default_Prob and Expected Loss using actual loan amounts
results = pd.DataFrame()
loan_amounts_test = original_loan_amounts.iloc[X_test_original.index]
models = {
    'Logistic_Regression': (logistic_regression_model, True),
    'Decision_Tree': (decision_tree_model, False),
    'Random_Forest': (random_forest_model, False)
}

for model_name, (model, use_scaler) in models.items():
    Default_Prob, expected_loss = predict_expected_loss(model, X_test_original, loan_amounts_test, use_scaler)
    results[f'Default_Prob_{model_name}'] = Default_Prob
    results[f'Expected_Loss_{model_name}'] = expected_loss
    print(f"Average Expected Loss per Loan for {model_name}: {np.mean(expected_loss)}")

# ROC Curve
plt.figure(figsize=(10, 8))
for model_name, (model, use_scaler) in models.items():
    if use_scaler:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test_original)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test_original, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test_original, y_pred_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend(loc='lower right')
plt.savefig('Quant_Research/roc_curve.png')
plt.show()

# Feature Importance
log_reg_feature_importance = np.abs(logistic_regression_model.coef_[0])
dt_feature_importance = decision_tree_model.feature_importances_
rf_feature_importance = random_forest_model.feature_importances_
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.barh(X_train_df.columns, log_reg_feature_importance, color='blue')
plt.title('Logistic Regression Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.subplot(3, 1, 2)
plt.barh(X_train_df.columns, dt_feature_importance, color='skyblue')
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.subplot(3, 1, 3)
plt.barh(X_train_df.columns, rf_feature_importance, color='green')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.tight_layout()
plt.savefig('Quant_Research/feature_importance_combined.png')
plt.show()

# Append original loan data to the results
output_data = pd.concat([loan_data.iloc[X_test_original.index].reset_index(drop=True), results.reset_index(drop=True)], axis=1)

# Save the full dataset with Default_Prob and expected loss for each model
output_path_full = 'Quant_Research/full_dataset_with_default_prob_and_expected_loss.csv'
output_data.to_csv(output_path_full, index=False)

print(f"Full dataset saved to {output_path_full}")
