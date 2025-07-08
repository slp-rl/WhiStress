import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import List, Union, Any, Dict


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    forced_decoder_ids: int
    eos_token_id: int
    transcription_column_name: str

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate and prepare batch for training or inference.
        
        This method handles:
        1. Padding audio features to the same length
        2. Identifying and preparing Whisper transcription labels
        3. Preparing emphasis detection head labels
        4. Masking special tokens for loss calculation
        5. Building a complete batch with all necessary tensors
        
        Args:
            features: List of feature dictionaries, each containing input features,
                      transcription labels, and emphasis labels
                      
        Returns:
            Dictionary containing padded tensors ready for model input:
            - input_features: Padded audio features
            - whisper_labels: Padded token IDs for transcription (with -100 for padding)
            - labels_head: Padded binary vectors for emphasis detection (with -100 for padding)
            - sentence_index: Tensor of sentence indices
        """
        # Step 1: Extract and pad the audio input features
        input_features_key = [elem for elem in list(features[0].keys()) if "input_features" in elem][0]
        input_features = [
            {"input_features": feature[input_features_key]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # Step 2: Identify the correct label column for Whisper transcription labels
        whisper_labels_key = 'whisper_labels'
        whisper_labels_key_opts = [elem for elem in list(features[0].keys()) if "labels" in elem and not "head" in elem and self.transcription_column_name in elem]
        if whisper_labels_key_opts != []:
            whisper_labels_key = whisper_labels_key_opts[0]
        if len(whisper_labels_key_opts) > 1:
            raise ValueError(
                f"More than one whisper_labels (backbone model labels) candidate found in features: {whisper_labels_key_opts}"
            )
            
        # Step 3: Extract and pad the transcription label sequences
        labels = [
            {"input_ids": feature[whisper_labels_key]} for feature in features
        ]
        # pad the labels to max length
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )

        # Step 4: Identify the column name of the correct label for emphasis detection head labels
        labels_head_key = 'labels_head'
        labels_head_key_opts = [elem for elem in list(features[0].keys()) if "labels_head" in elem and self.transcription_column_name in elem]
        if labels_head_key_opts != []:
            labels_head_key = labels_head_key_opts[0]
        if len(labels_head_key_opts) > 1:
            raise ValueError(
                f"More than one labels_head (added decoder head labels) candidate found in features: {labels_head_key_opts}"
            )
        # Handle labels_head (custom head labels)
        labels_head = [
            {"labels_head": feature[labels_head_key]} for feature in features
        ]
        max_len = max(
            [len(f["labels_head"]) for f in labels_head]
        )  # Find max length
        labels_head = torch.stack(
            [
                F.pad(
                    f["labels_head"], (0, max_len - len(f["labels_head"])), value=-100
                )
                for f in labels_head
            ]
        )

        # Step 5: Mask special tokens in labels_head
        ignore_tokens = [
            self.decoder_start_token_id,
            self.forced_decoder_ids,
            self.eos_token_id,
        ]

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        labels_head = torch.where(
            torch.isin(labels, torch.tensor(list(ignore_tokens))),
            torch.tensor(-100),
            labels_head,
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            labels_head = labels_head[:, 1:]

        # Step 6: Build the final batch dictionary
        batch['whisper_labels'] = labels
        batch['labels_head'] = labels_head
        batch["sentence_index"] = torch.tensor([feature["sentence_index"] for feature in features])
        assert batch['labels_head'].shape == batch['whisper_labels'].shape

        return batch
