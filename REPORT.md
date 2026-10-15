# Method report
**Problem.** Multi-class classification: 7 soil/weather features -> one of 22 crops.
**Data.** Kaggle Crop Recommendation dataset (2200 rows). Cleaning: missing values and duplicates are checked and removed; outliers are counted with the IQR rule but kept, because extreme values (e.g. high K for grapes/apple) are valid for some crops.
**Preprocessing.** `StandardScaler` inside a scikit-learn `Pipeline`, so it is fitted only on training folds (no leakage).
**Models.** Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, Gradient Boosting; 80/20 stratified split, 5-fold stratified CV on the training set. Metrics: accuracy, macro precision/recall/F1, 22x22 confusion matrix, classification report (`models/`, `figures/`).
**Tuning.** The model with the best CV accuracy is tuned with GridSearchCV and saved with joblib. Feature importance uses permutation importance on the test set (works for any model).
**Advice rules.** Each crop has a typical range (10th-90th percentile of its rows). N, P, K and rainfall below/above the best crop's range trigger fertilizer, irrigation or drainage advice; pH < 5.5 suggests lime, pH > 7.5 gypsum/organic matter.
**Limits.** Estimates from a small, clean dataset; no yield, price, region or season. Consult a local agriculture officer.
**Future scope.** Live weather API, soil-lab integration, yield and price prediction, more crops and regional languages.
