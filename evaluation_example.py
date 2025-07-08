import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from whistress import WhiStressInferenceClient

CURRENT_DIR = Path(__file__).parent

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

def compute_prf_metrics(predictions, references, average="binary"):
    """
    Computes precision, recall, and F1 using Hugging Face's `evaluate`.
    Args:
        predictions (List[int]): Model's predicted labels.
        references  (List[int]): True labels.
        average     (str): "binary", "macro", "micro", or "weighted".
                          Use "binary" for two-class tasks.
    Returns:
        Dict[str, float]: e.g. {"precision": 0.8, "recall": 0.75, "f1": 0.77}
    """
    p = precision_metric.compute(predictions=predictions, references=references, average=average)["precision"]
    r = recall_metric.compute(predictions=predictions, references=references, average=average)["recall"]
    f = f1_metric.compute(predictions=predictions, references=references, average=average)["f1"]

    return {"precision": p, "recall": r, "f1": f}


def calculate_metrics_on_dataset(dataset):
    """
    Sample structure example for slprl/StressTest dataset:
    # {'transcription_id': '98dd4a46-6b59-4817-befc-e35d548465c7',
    #  'transcription': 'You chose to do this?',
    #  'description': 'You voluntarily do this without being forced?',
    #  'intonation': 'You *chose* to do this?',
    #  'interpretation_id': '8dbe4033-2451-4674-bf9d-b8e05c61e9c4',
    #  'audio': {'path': None,
    #   'array': array([-1.22070312e-04, -9.15527344e-05, -6.10351562e-05, ...,
    #           2.44140625e-04,  2.13623047e-04,  2.44140625e-04]),
    #   'sampling_rate': 16000},
    #  'metadata': {'gender': 'male',
    #   'language_code': 'en',
    #   'sample_rate_hertz': 16000,
    #   'voice_name': 'actor'},
    #  'possible_answers': ['Why did you choose this option out of all of them?',
    #   'You voluntarily do this without being forced?'],
    #  'label': 1,
    #  'stress_pattern': {'binary': [0, 1, 0, 0, 0],
    #   'indices': [1],
    #   'words': ['chose']}
    """
    predictions = []
    references = []
    for sample in tqdm(dataset):
        gt_stresses = sample['stress_pattern']['binary']
        # scored = 
        scored = whistress_client.predict(
            audio=sample['audio'],
            # Using ground truth transcription for evaluating stress prediction ability. 
            # set transcription to None if not available
            transcription=sample['transcription'], 
            return_pairs=True
        )
        _, pred_stresses = zip(*scored)
        # Ensure the lengths are the same 
        # When transcription is not provided, predictions should be aligned with the ground truth
        assert len(pred_stresses) == len(gt_stresses), "Length mismatch"
        predictions.extend(pred_stresses)
        references.extend(gt_stresses)
        
    return compute_prf_metrics(predictions, references, average="binary")


if __name__ == "__main__":
    from datasets import load_dataset
    import json
    # Load your dataset, replace with the actual dataset you are using
    dataset_name = "slprl/StressTest"  # Example dataset name, change as needed
    dataset = load_dataset(dataset_name)
    split_name = 'test' 
    print(f"Evaluating WhiStress on {dataset_name} for split {split_name}...")
    metrics = calculate_metrics_on_dataset(dataset=dataset[split_name])
    # Save or log the metrics as needed
    results = {
        "dataset": dataset_name,
        "split": split_name,
        "metrics": metrics
    }
    print(f"Results: {results}")
    results_dir = CURRENT_DIR / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/whistress_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
        