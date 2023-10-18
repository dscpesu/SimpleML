# Comparative study of classification models

### Random Forest Classifier:
The Random Forest Classifier's high performance can be attributed to the following features:
* High accuracy and robust performance
* Less prone to overfitting compared to individual decision trees
* Automatically handles feature importance

### Artificial Neural Network(ANN): 
ANN can capture complex relationships in data and hence it's good performance. However it is prone to overfitting.

### Support Vector Machines(SVM):
SVM is versatile(works well for both linear and non-linear data) and is robust against overfitting in high-dimensional sapce. However it can be sensitive to the choice of the kernel and interpretability may be challenging.

### Decision Tree:
Decision trees are simple to understand and interpret and require minimal data preparation. However they are prone to overfitting and lack robustness(small changes in data can lead to different tree structures).

### Logistic Regressor:
Logistic Regressor is efficient for small to medium-sized datasets. However it assumes a linear decision boundary and is sensitive to outliers. It is also limited in capturing complex relationships.

### Stochastic Gradient Descent(SGD):
SGD is efficient for large datasets. However it requires careful tuning of hyperparameters, explaining its poor performance.

### KNN Classifier:
KNN Classifier is fast for small datasets(No training phase) and is non-parametric and suitable for non-linear relationships. However it is sensitive to irrelevant features and noise and performance degrades with high-dimensional data. Hence its performance was poor.

### Naive Bayes:
Naive Bayes performs well with small datasets and handles high-dimensional data. However it assumes independence between features and it is sensitive to irrelevant features. It may not capture complex relationships. Hence it performed poorly with the given dataset.