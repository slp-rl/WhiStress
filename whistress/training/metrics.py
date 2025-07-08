import evaluate


class WhiStressMetrics:
    def __init__(self):
        self.evaluation_metrics = {
            "accuracy": evaluate.load("accuracy"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1"),
        }

    def compute_metrics(self, pred):

        def ignore_masked_predictions(pred_ids, label_ids, pad_token_id):
            # Create a mask where label_ids is not equal to pad_token_id
            mask_label_ids = label_ids != pad_token_id

            # Flatten the tensors to process them as one-dimensional arrays
            pred_ids_flat = pred_ids[mask_label_ids].flatten()
            label_ids_flat = label_ids[mask_label_ids].flatten()

            return pred_ids_flat, label_ids_flat

        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        preds, labels = ignore_masked_predictions(pred_ids, label_ids, -100)

        metrics = {}
        metrics.update(
            self.evaluation_metrics["accuracy"].compute(
                predictions=preds, references=labels
            )
        )
        metrics.update(
            self.evaluation_metrics["precision"].compute(
                predictions=preds, references=labels, pos_label=1, zero_division=0
            )
        )
        metrics.update(
            self.evaluation_metrics["recall"].compute(
                predictions=preds, references=labels, pos_label=1, zero_division=0
            )
        )
        metrics.update(
            self.evaluation_metrics["f1"].compute(
                predictions=preds, references=labels, pos_label=1
            )
        )

        return metrics
