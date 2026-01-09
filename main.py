from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.models.train_models import train_multiple_models
from src.models.evaluate_models import evaluate_models
from src.visualization.plot_utils import plot_roc_curve
import pandas as pd
# Load the data
df = load_data('./data/raw/creditcard_2023.csv')

# Preprocess the data
X = df.drop(columns=['Class'])
y = df['Class']
X_resampled, y_resampled = preprocess_data(X, y)

# Train models
models = train_multiple_models(X_resampled, y_resampled)

# Evaluate models
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
evaluation_results = evaluate_models(models, X_test, y_test)

# Plot ROC curves
plot_roc_curves(models, X_test, y_test)