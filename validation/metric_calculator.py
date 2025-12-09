#from sklearn.metrics import roc_auc_score

class MetricCalculator:
    def compute(self, results):# predictions, targets):
        # Convert YOLO predictions → probability scores per class
        #y_true, y_score = self._prepare_auc_inputs(predictions, targets)
        auc = 0.5
       # auc = roc_auc_score(y_true, y_score)

        return {
            "auc": auc
        }

    def _prepare_auc_inputs(self, predictions, targets):
        """
        Convert YOLO output into AUC-compatible vectors.
        Returns:
            y_true: ground truth labels
            y_score: predicted probabilities or scores
        """
        y_true = []
        y_score = []

        for p, t in zip(predictions, targets):
            y_true.append(t["label"])
            y_score.append(p["score"])

        return y_true, y_score
