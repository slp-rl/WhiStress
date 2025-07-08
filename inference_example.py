import torch
from datasets import load_dataset
from whistress import WhiStressInferenceClient


if __name__ == "__main__":
    
    # Load your dataset, replace with the actual dataset you are using
    dataset_name = "slprl/StressTest"  # Example dataset name, change as needed
    dataset = load_dataset(dataset_name)
    split_name = 'test'

    sample = dataset[split_name][0]
    """
    Sample structure example for slp-rl/StressTest dataset:
    # {'transcription_id': '50d0ae80-b5bb-43f8-a43f-134f8c4b0ac4',
    #  'transcription': 'I didn't say he stole the money.',
    #  'description': 'Someone else stole it, not him.',
    #  'intonation': 'I didn't say *he* stole the money.',
    #  'interpretation_id': '9c1a7c2f-6980-4d7a-b6a7-9a0c7f464104',
    #  'audio': {'path': None,
    #   'array': array([-1.22070312e-04, -9.15527344e-05, -6.10351562e-05, ...,
    #           2.44140625e-04,  2.13623047e-04,  2.44140625e-04]),
    #   'sampling_rate': 16000},
    #  'stress_pattern': {'binary': [0, 0, 0, 1, 0, 0, 0], 'indices': [3], 'words': ['he']}
    """
    print(f'GT transcription: {sample["transcription"]}')
    print(f'GT stressed words: {sample["stress_pattern"]}')

    print("Loading WhiStress model for inference...")
    whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

    pred_transcription, pred_stresses = whistress_client.predict(
            audio=sample['audio'],
            # Using whisper transcription for inference, inference from audio only.
            transcription=None, 
            return_pairs=False
        )
    print(f'Predicted transcription: {pred_transcription}')
    print(f'Predicted stressed words: {pred_stresses}')

