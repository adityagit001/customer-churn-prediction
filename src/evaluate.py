import pandas as pd


def _binary_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, and F1 for binary labels."""
    true = pd.Series(y_true).reset_index(drop=True)
    pred = pd.Series(y_pred).reset_index(drop=True)
    actual_positive = true == 1
    predicted_positive = pred == 1

    tp = (actual_positive & predicted_positive).sum()
    fp = (~actual_positive & predicted_positive).sum()
    fn = (actual_positive & ~predicted_positive).sum()

    accuracy = (true == pred).mean()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return accuracy, precision, recall, f1


def _roc_auc(y_true, scores):
    """Calculate ROC-AUC using average ranks, including tied scores."""
    true = pd.Series(y_true).reset_index(drop=True)
    scores = pd.Series(scores).reset_index(drop=True)
    positive = true == 1
    n_positive = positive.sum()
    n_negative = (~positive).sum()
    if not n_positive or not n_negative:
        return None

    ranks = scores.rank(method="average")
    return ((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative))


def evaluate_models(trained_models, X_test, y_test):
    results = []

    for name, model in trained_models.items():

        # Predictions
        y_pred = model.predict(X_test)

        # Probability for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None

        # Metrics
        acc, prec, rec, f1 = _binary_metrics(y_test, y_pred)

        if y_proba is not None:
            roc = _roc_auc(y_test, y_proba)
        else:
            roc = None

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4) if roc is not None else None
        })

    results_df = pd.DataFrame(results)

    # Sort only if ROC-AUC values exist
    if "ROC-AUC" in results_df.columns:
        results_df = results_df.sort_values(
            by="ROC-AUC",
            ascending=False,
            na_position="last"
        )

    return results_df