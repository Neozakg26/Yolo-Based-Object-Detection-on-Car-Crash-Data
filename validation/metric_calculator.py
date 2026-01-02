#from sklearn.metrics import roc_auc_score


## NOT USED FOR NOW. CREATED incase Results from YOLO val aren't enough 
## TO BE DELETED IF final decision  is yolo.val is enough. 
class MetricCalculator:
    @staticmethod
    def compute(results): # predictions, targets):
        # Convert YOLO predictions → probability scores per class
        #y_true, y_score = self._prepare_auc_inputs(predictions, targets)
        auc = 0.5
       # auc = roc_auc_score(y_true, y_score)
        results = results
        return {
            "auc": auc
        }
    @staticmethod
    def _prepare_auc_inputs( predictions, targets):
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
