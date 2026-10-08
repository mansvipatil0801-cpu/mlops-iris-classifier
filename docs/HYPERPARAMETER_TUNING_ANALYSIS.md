# Hyperparameter Tuning Analysis

## 1. Baseline Model

Model:

DecisionTreeClassifier

CV F1 Macro:

0.9663

Test Accuracy:

0.9000

## 2. Grid Search

Model:

RandomForestClassifier

Total combinations:

72

Cross-validation:

5-fold

Total fits:

360

Best Parameters:

- max_depth: 3
- max_features: sqrt
- min_samples_split: 2
- n_estimators: 50

Best CV F1 Macro:

0.9663

Test Accuracy:

0.9667

## 3. Random Search

Model:

RandomForestClassifier

Number of iterations:

30

Cross-validation:

5-fold

Total fits:

150

Best Parameters:

- max_depth: 3
- max_features: sqrt
- min_samples_split: 6
- n_estimators: 100

Best CV F1 Macro:

0.9663

Test Accuracy:

0.9667

## 4. Comparison

Grid Search evaluated all 72 combinations from the specified
hyperparameter grid.

Random Search evaluated 30 selected combinations from the
hyperparameter search space.

Grid Search required 360 model fits, while Random Search
required only 150 model fits.

The baseline Decision Tree achieved 0.9000 test accuracy.

Both Grid Search and Random Search achieved 0.9667 test accuracy,
which is higher than the baseline.

Grid Search and Random Search achieved the same best CV F1 Macro
score of 0.9663 in this experiment.

Therefore, Random Search was more computationally efficient because
it achieved the same CV F1 Macro and test accuracy as Grid Search
using only 150 fits instead of 360.