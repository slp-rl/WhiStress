from transformers import Seq2SeqTrainer
from tqdm import tqdm
import torch
import numpy as np
import os
import json


class WhiStressTrainer(Seq2SeqTrainer):
    """
    Custom trainer extending Seq2SeqTrainer for speech emphasis detection.
    
    Implements specialized training, evaluation, and model saving methods
    designed specifically for the emphasis detection model architecture.
    """

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        Pad tensors to a specified maximum length using -100 as padding token.
        
        Args:
            tensor: Input tensor to pad
            max_length: Target length for padded tensor
            
        Returns:
            Padded tensor of shape (batch_size, max_length)
        """
        pad_token_id = -100

        # Create a padded tensor using the custom pad token
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )

        # Ensure that the tensor fits within the padded tensor up to the original tensor's length
        padded_tensor[:, : tensor.shape[-1]] = tensor

        return padded_tensor

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Execute a single training step with gradient clipping.
        
        Removes sentence indices from inputs before passing to parent class,
        then applies gradient clipping to prevent exploding gradients.
        
        Args:
            model: Model to train
            inputs: Dictionary of input tensors
            num_items_in_batch: Optional parameter specifying batch size
            
        Returns:
            Loss value for the training step
        """
        sentence_index = inputs.pop("sentence_index")
        # Perform the default training step
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Clip gradients manually
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        return loss

    def save_final_model(self, output_dir=None, training_args=None):
        """
        Save only the emphasis detection components of the model.
        
        Rather than saving the entire model, this method saves only:
        1. The classifier (head) used for emphasis detection
        2. The additional decoder block
        3. The selected layer passed to the head
        4. Training arguments for reproducibility
        
        Args:
            output_dir: Directory to save model components
            training_args: Training arguments to save
        """
        # save only the classifier and extra decoder layer
        classifier = (
            self.model.classifier if hasattr(self.model, "classifier") else None
        )
        additional_decoder_block = (
            self.model.additional_decoder_block
            if hasattr(self.model, "additional_decoder_block")
            else None
        )
        if output_dir is not None:
            torch.save(
                classifier.state_dict(), os.path.join(output_dir, "classifier.pt")
            )
            torch.save(
                additional_decoder_block.state_dict(),
                os.path.join(output_dir, "additional_decoder_block.pt"),
            )
            # save the layer passed to the head
            layer_for_head = self.model.layer_for_head
            with open(os.path.join(output_dir, "metadata.json"), "w") as file:
                json.dump({"layer_for_head": layer_for_head}, file)
            # save the training arguments
            with open(os.path.join(output_dir, "training_args.json"), "w") as file:
                json.dump(training_args.to_dict(), file)
            
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at token level on the evaluation dataset.
        
        Runs a forward pass through the model for each batch, collects predictions
        and labels, and calculates evaluation metrics. Operates at the token level,
        meaning each token's emphasis prediction is evaluated separately.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][0]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][0]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                """uncomment the following block if you want to use the generate method"""
                # generated_ids = self.model.generate(
                #     input_features=input_features,
                #     # decoder_input_ids=batch.get("decoder_input_ids"),  # If required
                #     max_length=self.args.generation_max_length,
                #     # labels_head=None,
                #     whisper_labels=whisper_labels,
                #     # **self.args.generation_kwargs
                # )
                # Run forward pass through the model to get predictions 
                # (assuming whisper labels are aligned in length with labels_head)
                # Instead of using generate(), we use a direct forward pass which is faster
                # and returns logits that can be converted to binary predictions
                generated_ids = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels
                )['preds']  # Extract the 'preds' field which contains binary predictions
                
            # Pad predictions and labels to the same length for proper comparison
            # This ensures all tensors in the batch have consistent dimensions
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            
            # Convert to CPU and numpy for accumulation
            # We collect all batch predictions before computing metrics
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Concatenate all batches into single arrays for metric computation
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute evaluation metrics using the provided compute_metrics function
        # This typically calculates precision, recall, F1, and other relevant metrics
        outputs_metrics = {}
        if self.compute_metrics is not None:
            # The compute_metrics function expects a dictionary with predictions and labels
            metrics = self.compute_metrics(
                {"predictions": all_preds, "label_ids": all_labels}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
                    
        with open(os.path.join(self.args.output_dir, "log_eval.txt"), "a") as file:
            json.dump(f'Evaluate at TOKEN LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)
        # print(f'{eval_dataset_name} : {outputs_metrics}')
        return outputs_metrics
    
    def evaluate_at_word_level(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at word level on the evaluation dataset.
        
        Similar to evaluate(), but aggregates token-level predictions to the word level
        before computing metrics. This provides a more meaningful evaluation for emphasis
        detection since emphasis typically applies to entire words, not individual tokens.
        
        A word is considered emphasized if any of its tokens are predicted as emphasized.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of word-level evaluation metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds_by_words = []
        all_labels_by_words = []

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][-1]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][-1]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                # generated_ids = self.model.generate(
                #     input_features=input_features,
                #     # decoder_input_ids=batch.get("decoder_input_ids"),  # If required
                #     max_length=self.args.generation_max_length,
                #     # labels_head=None,
                #     whisper_labels=whisper_labels,
                #     # **self.args.generation_kwargs
                # )
                generated_ids = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels,
                )['preds']
                
            all_labels_head_by_words = []
            all_generated_ids_by_words = []
            batch_samples = torch.where(eval_dataset['sentence_index'].cpu() == batch['sentence_index'].unsqueeze(1).cpu())[1].numpy()
            map_dict_key = [elem for i, elem in enumerate(eval_dataset.column_names) if "map_dict" in elem][-1]
            for i in range(labels_head.shape[0]):
                j_start = 1
                labels_head_by_words = [-100]
                generated_ids_by_words = [-100]
                for val in eval_dataset[int(batch_samples[i])][map_dict_key]["values"]:
                    if len(val) == 0:
                        j_end += 1
                    else:
                        j_end = j_start + len(val)
                    while (not np.array_equal(whisper_labels[i][j_start:j_end].cpu().numpy(), val.numpy()) and \
                        not whisper_labels[i][j_end].item() == 50256 and not len(val) == 0):
                        # if we ran into tokens which aren't part of a word (like ',', '.', '\n'), we skip them/treat them as a word
                        # the second condition in the while loop is meant to prevent crossing the end of sequence
                        if 1 in labels_head[i][j_start]:
                            labels_head_by_words.append(1)
                        else:
                            labels_head_by_words.append(0)
                        if 1 in generated_ids[i][j_start]:
                            generated_ids_by_words.append(1)
                        else:
                            generated_ids_by_words.append(0)
                        j_start += 1
                        j_end += 1
                    if 1 in labels_head[i][j_start:j_end]:
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if 1 in generated_ids[i][j_start:j_end]:
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_start = j_end
                # add the last punctuation mark if it's not the end of the sequence
                if whisper_labels[i][j_end].item() != 50256: # 50256 relates to the choice of the backbone as whisper's english model
                    if 1 in labels_head[i][j_end]:
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if 1 in generated_ids[i][j_end]:
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_end += 1
                assert labels_head[i][j_end]==-100
                    
                labels_head_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(labels_head_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )
                generated_ids_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(generated_ids_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )                
                all_generated_ids_by_words.append(generated_ids_by_words_padded.squeeze(0))
                all_labels_head_by_words.append(labels_head_by_words_padded.squeeze(0))
                
            padded_labels_by_words = torch.stack(all_labels_head_by_words)
            padded_preds_by_words = torch.stack(all_generated_ids_by_words)
            
            all_preds_by_words.append(padded_preds_by_words.cpu().numpy())
            all_labels_by_words.append(padded_labels_by_words.cpu().numpy())

        # Flatten lists        
        all_preds_by_words = np.concatenate(all_preds_by_words, axis=0)
        all_labels_by_words = np.concatenate(all_labels_by_words, axis=0)

        # Compute metrics
        outputs_metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                {"predictions": all_preds_by_words, "label_ids": all_labels_by_words}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
                    
        with open(os.path.join(self.args.output_dir, "log_eval_word_level.txt"), "a") as file:
            json.dump(f'Evaluate at WORD LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)
        return outputs_metrics

    def align_samples_aux(self, pred):
        """
        Identify samples where predictions and labels have mismatched lengths.
        
        Used to filter out problematic samples where the model's predictions
        cannot be directly compared to ground truth labels due to length mismatch.
        
        Args:
            pred: Dictionary containing 'predictions' and 'label_ids' arrays
            
        Returns:
            List of row indices to remove from evaluation
        """
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        pad_token_id = -100

        rows_to_remove = []
        for i, (pred_id, label_id) in enumerate(zip(pred_ids, label_ids)):
            # Create a mask where pred_ids are not equal to pad_token_id
            mask_pred_ids = pred_id != pad_token_id
            # Create a mask where label_ids are not equal to pad_token_id
            mask_label_ids = label_id != pad_token_id
            if pred_id[mask_pred_ids].shape[0] != label_id[mask_label_ids].shape[0]:
                rows_to_remove.append(i)

        return rows_to_remove
    
    def aligned_whisper_transcriptions(self, example):
        """
        Generate Whisper transcriptions and check alignment with ground truth.
        
        Used during dataset preprocessing to identify samples where the Whisper model's
        transcription matches the ground truth transcription, ignoring formatting
        differences like capitalization and punctuation.
        
        Args:
            example: Dataset example containing audio and transcription
            
        Returns:
            Example with added 'aligned_whisper_transcriptions' field
        """
        # Filter out samples with '\n' in the transcription
        token_ids = self.model.whisper_model.generate(input_features=example['input_features'].to('cuda').unsqueeze(0), 
                                                    labels=example['whisper_labels'].to('cuda').unsqueeze(0))
        transcription = self.model.processor.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        example['aligned_whisper_transcriptions'] = ''
        if transcription.lstrip().lower().replace(',','').replace('.','') == example['transcription'].lower():
            example['aligned_whisper_transcriptions'] = transcription
        return example
    
    def filter_misaligned_samples(self, example):
        """
        Filter out examples where Whisper transcription doesn't align with ground truth.
        
        Args:
            example: Dataset example containing aligned_whisper_transcriptions
            
        Returns:
            Boolean indicating whether the example should be kept (True) or filtered out (False)
        """
        return example['aligned_whisper_transcriptions'] != ''

    def align_samples(self, dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Process dataset to identify and flag misaligned samples.
        
        Generates model predictions for each example in the dataset, and identifies
        examples where the prediction length doesn't match the label length, which
        would cause evaluation errors.
        
        Args:
            dataset: Dataset to check for alignment issues
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            
        Returns:
            List of indices for samples that should be removed due to alignment issues
        """
        eval_dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()

        all_preds = []
        all_labels = []

        for i, batch in enumerate(tqdm(eval_dataloader)):
            # Extract input features and labels
            input_features = batch["input_features"]
            whisper_labels = batch["whisper_labels"]
            labels_head = batch["labels_head"]

            # Generate predictions
            with torch.no_grad():
                # Adjust inputs according to your model's requirements
                generated_ids = self.model.generate(
                    input_features=input_features,
                    whisper_labels=whisper_labels,
                )

            # Pad or truncate predictions and labels to a fixed length
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            # Collect predictions and labels
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Flatten lists
        for i in range(len(all_preds)):
            print(f"{all_preds[i].shape=}, {all_labels[i].shape=}")
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return self.align_samples_aux(
            {"predictions": all_preds, "label_ids": all_labels}
        )
