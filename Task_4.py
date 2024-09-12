import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

file_path = 'Quant_Research/Task 3 and 4_Loan_Data.csv'
loan_data = pd.read_csv(file_path)

fico_scores = loan_data['fico_score'].values
defaults = loan_data['default'].values

n_buckets = 5

## MSE Minimization using KMeans
def mse_minimization(fico_scores, n_buckets):
    fico_scores_reshaped = fico_scores.reshape(-1, 1)
    kmeans = KMeans(n_clusters=int(round(n_buckets)), random_state=42)
    kmeans.fit(fico_scores_reshaped)
    bucket_labels = kmeans.labels_
    bucket_centers = kmeans.cluster_centers_.flatten()
    bucketed_fico_scores = bucket_centers[bucket_labels]
    return bucket_labels, bucketed_fico_scores

### Log-Likelihood Maximization
def dynamic_log_likelihood_maximization(fico_scores, defaults, n_buckets):
    boundaries = np.linspace(fico_scores.min(), fico_scores.max(), n_buckets + 1)
    
    def log_likelihood(boundaries, fico_scores, defaults, n_buckets):
        log_likelihood = 0
        for i in range(n_buckets):
            in_bucket = (fico_scores >= boundaries[i]) & (fico_scores < boundaries[i + 1])
            n_i = np.sum(in_bucket)
            k_i = np.sum(defaults[in_bucket])
            if n_i == 0:
                continue
            p_i = k_i / n_i if n_i != 0 else 0
            if p_i > 0 and p_i < 1:
                log_likelihood += k_i * np.log(p_i) + (n_i - k_i) * np.log(1 - p_i)
        return -log_likelihood
    
    result = minimize(log_likelihood, boundaries, args=(fico_scores, defaults, n_buckets), method='Nelder-Mead')
    optimized_boundaries = result.x
    return optimized_boundaries

# assign FICO scores to buckets
def assign_buckets_log_likelihood(fico_scores, boundaries):
    bucket_labels = np.zeros(fico_scores.shape, dtype=int)
    for i in range(1, len(boundaries)):
        bucket_labels[(fico_scores >= boundaries[i-1]) & (fico_scores < boundaries[i])] = i - 1
    return bucket_labels

# Bayesian Optimization for MSE
def bayesian_optimization_mse():
    def mse_to_minimize(n_buckets):
        n = int(round(n_buckets))
        bucket_labels, bucketed_fico_scores = mse_minimization(fico_scores, n_buckets)
        mse_score = np.mean((fico_scores - bucketed_fico_scores) ** 2)
        return -mse_score

    optimizer = BayesianOptimization(f=mse_to_minimize, pbounds={'n_buckets': (2, 10)}, random_state=42)
    optimizer.maximize(init_points=5, n_iter=10)
    return int(optimizer.max['params']['n_buckets'])

# Agglomerative Clustering
def agglomerative_clustering(fico_scores, n_buckets):
    fico_scores_reshaped = fico_scores.reshape(-1, 1)
    agg_clustering = AgglomerativeClustering(n_clusters=n_buckets)
    bucket_labels = agg_clustering.fit_predict(fico_scores_reshaped)
    return bucket_labels

# Apply each bucketing technique
bucket_labels_mse, bucketed_fico_mse = mse_minimization(fico_scores, n_buckets)
optimized_boundaries_ll = dynamic_log_likelihood_maximization(fico_scores, defaults, n_buckets)
bucket_labels_ll = assign_buckets_log_likelihood(fico_scores, optimized_boundaries_ll)
best_n_buckets_bayesian = bayesian_optimization_mse()
bucket_labels_bayesian, _ = mse_minimization(fico_scores, best_n_buckets_bayesian)
bucket_labels_agg = agglomerative_clustering(fico_scores, n_buckets)

# Add bucket labels to the dataset
loan_data['Bucket_MSE'] = bucket_labels_mse
loan_data['Bucket_LL'] = bucket_labels_ll
loan_data['Bucket_Bayesian'] = bucket_labels_bayesian
loan_data['Bucket_Agg'] = bucket_labels_agg

# Evaluation

# AUC and precision-recall
def evaluate_bucketing_technique(bucket_labels, method_name):
    precision, recall, _ = precision_recall_curve(defaults, bucket_labels)
    auc_score = roc_auc_score(defaults, bucket_labels)
    fpr, tpr, _ = roc_curve(defaults, bucket_labels)

    return {
        'precision': precision,
        'recall': recall,
        'auc_score': auc_score,
        'fpr': fpr,
        'tpr': tpr
    }

# Apply evaluation
results_mse = evaluate_bucketing_technique(bucket_labels_mse, "MSE Minimization")
results_ll = evaluate_bucketing_technique(bucket_labels_ll, "Log-Likelihood Maximization")
results_bayesian = evaluate_bucketing_technique(bucket_labels_bayesian, "Bayesian Optimization")
results_agg = evaluate_bucketing_technique(bucket_labels_agg, "Agglomerative Clustering")

# Store results
evaluation_results = {
    'MSE Minimization': results_mse,
    'Log-Likelihood Maximization': results_ll,
    'Bayesian Optimization': results_bayesian,
    'Agglomerative Clustering': results_agg
}

# Visualization
plt.figure(figsize=(14, 6))
for method, results in evaluation_results.items():
    plt.plot(results['recall'], results['precision'], label=method)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Quant_Research/precision_recall_comparison.png')

plt.figure(figsize=(14, 6))
for method, results in evaluation_results.items():
    plt.plot(results['fpr'], results['tpr'], label=f"{method} (AUC={results['auc_score']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Quant_Research/roc_curve_comparison.png')

# Save
output_path_final = 'Quant_Research/bucketed_fico_scores_final.csv'
loan_data.to_csv(output_path_final, index=False)
print(f"Final dataset with bucketed FICO scores and all methods saved to {output_path_final}")
