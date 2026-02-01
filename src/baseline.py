# src/baseline.py
# Baseline model for churn prediction
# Just predicts the most frequent class (majority vote)

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)


class BaselineModel:
    """
    Simple baseline that always predicts the majority class.
    Used as a benchmark to compare against real models.
    """
    
    def __init__(self):
        self.most_frequent_class = None
        self.class_distribution = None
        self._is_fitted = False
    
    def fit(self, X, y):
        """Learn the most frequent class from training data."""
        y = np.array(y)
        
        unique, counts = np.unique(y, return_counts=True)
        self.class_distribution = dict(zip(unique, counts))
        self.most_frequent_class = unique[np.argmax(counts)]
        self._is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict majority class for all samples."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        n_samples = len(X)
        return np.full(n_samples, self.most_frequent_class)
    
    def predict_proba(self, X):
        """Return class probabilities based on training distribution."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        n_samples = len(X)
        total = sum(self.class_distribution.values())
        
        prob_0 = self.class_distribution.get(0, 0) / total
        prob_1 = self.class_distribution.get(1, 0) / total
        
        return np.full((n_samples, 2), [prob_0, prob_1])
    
    def evaluate(self, X, y_true):
        """Calculate metrics on given data."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def __repr__(self):
        if self._is_fitted:
            return f"BaselineModel(predicts={self.most_frequent_class})"
        return "BaselineModel(not fitted)"


def print_metrics(metrics, name=""):
    """Print metrics in readable format."""
    print(f"\n{name} Metrics:" if name else "\nMetrics:")
    print("-" * 35)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing BaselineModel")
    print("=" * 35)
    
    np.random.seed(42)
    X_test = np.random.randn(100, 5)
    y_test = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    
    model = BaselineModel()
    model.fit(X_test, y_test)
    
    print(f"Model: {model}")
    print(f"Class counts: {model.class_distribution}")
    
    metrics = model.evaluate(X_test, y_test)
    print_metrics(metrics, "Test")
