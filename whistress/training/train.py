import torch
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import WhisperConfig, Seq2SeqTrainingArguments, TrainerCallback, set_seed
from ..model.model import WhiStress
from .data_loader import load_data
from .data_collator import DataCollatorSpeechSeq2SeqWithPadding
from .trainer import WhiStressTrainer
from .metrics import WhiStressMetrics

CURRENT_DIR = Path(__file__).parent
WANDB_API_KEY = os.environ.get("WAND_API_KEY", None)

class CustomCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # Example of handling serialization
        if hasattr(state, "metrics") and isinstance(state.metrics, np.ndarray):
            state.metrics = state.metrics.tolist()

def train_or_evaluate(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    whisper_backbone_name = f"openai/whisper-small.en"
    whisper_config = WhisperConfig()
    layer_for_head = 9
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        whistress_model = WhiStress(
            whisper_config, layer_for_head=layer_for_head, whisper_backbone_name=whisper_backbone_name
        ).to(device)
        whistress_model.load_model(args.model_path, device=device)
        whistress_model.to(device)
        whistress_model.eval()
    else:
        logger.info("Training a new model from scratch")
        whistress_model = WhiStress(
            whisper_config, layer_for_head=layer_for_head, whisper_backbone_name=whisper_backbone_name
        ).to(device)
    
    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=whistress_model.processor,
        decoder_start_token_id=whistress_model.whisper_model.config.decoder_start_token_id,
        forced_decoder_ids=whistress_model.whisper_model.config.forced_decoder_ids[
            0
        ][1],
        eos_token_id=whistress_model.whisper_model.config.eos_token_id,
        transcription_column_name=args.transcription_column_name
    )

    train, val = None, None
    if args.is_train:
        DatasetTrain = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_train, save_path=args.dataset_path)
        train, val, _ = DatasetTrain.split_train_val_test()
    DatasetEval = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_eval, save_path=args.dataset_path)
    _, _, test = DatasetEval.split_train_val_test()
    
    print(f"Output path for the training run: {args.output_path}")
    output_path = args.output_path

    if WANDB_API_KEY and args.is_train:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project="whistress",
            name=f"{args.dataset_train}_{args.dataset_eval}",
            config={
                "dataset_train": args.dataset_train,
                "dataset_eval": args.dataset_eval,
                "transcription_column_name": args.transcription_column_name,
            },
            dir=output_path,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,  # change to a repo name of your choice
        # per_device_train_batch_size=4, # assuming 8 gpus. decrease to ~2 for small dataset, increase to ~4 for large dataset.
        per_device_train_batch_size=32, # assuming 1 gpu. 
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-4, # decrease to ~5e-4 for large dataset, increase to ~4e-4 for small dataset
        warmup_ratio=0.05,
        num_train_epochs=2, # Increase to 4 for task-specific evaluation, use 2 for zero-shot - More generalized evaluation. 
        seed=42,
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        # per_device_eval_batch_size=4,
        per_device_eval_batch_size=32,
        generation_max_length=96,
        save_steps=5000,
        eval_steps=10,
        logging_steps=10,
        weight_decay=0.01,
        push_to_hub=False,
        report_to=['wandb'] if WANDB_API_KEY is not None else None,
        label_names=[f"labels_head_{args.transcription_column_name}", "sentence_index", f"labels_{args.transcription_column_name}"],
        overwrite_output_dir=True, # change if you want to keep previous output
    )
    # Set seed before initializing model.
    set_seed(training_args.seed)
    metrics = WhiStressMetrics()

    trainer_emphasis = None
    if args.is_train:
        trainer_emphasis = WhiStressTrainer(
        args=training_args,
        model=whistress_model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        compute_loss_func=whistress_model.loss_fct,
        processing_class=whistress_model.processor.feature_extractor,
        )
        trainer_emphasis.evaluate_at_word_level(
            # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args.dataset_eval}-initial-word_level",
        )
        trainer_emphasis.evaluate(
            # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args.dataset_eval}-initial",
        )
        trainer_emphasis.train()
        # trainer_emphasis.save_model(trainer_emphasis.args.output_dir)
        trainer_emphasis.save_final_model(args.output_path, training_args)
    else:
        # change the trainer for evaluation only
        trainer_emphasis = WhiStressTrainer(
            args=training_args,
            model=whistress_model,
            train_dataset=test, # we don't really use it, but the trainer requires it
            eval_dataset=test, # we don't really use it, but the trainer requires it
            data_collator=data_collator,
            compute_metrics=metrics.compute_metrics,
            tokenizer=whistress_model.processor.feature_extractor,
        )

    trainer_emphasis.evaluate_at_word_level(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args.dataset_eval}-final-word_level",
    )
    trainer_emphasis.evaluate(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args.dataset_eval}-final",
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Main function to execute the training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model to be loaded for evaluation.\
                If training, a model path (of the final model) must not be provided",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset directory to save to or load the preprocessed dataset from.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the output directory for the training run. \
                If not provided, a new directory named training_results will be created under the current directory.",
    )
    parser.add_argument(
        "--transcription_column_name",
        type=str,
        choices=["transcription", "aligned_whisper_transcriptions"],
        default="transcription",
        help="""Name of the transcription column in the dataset. 
        transcription: The original transcription text as written from the raw dataset.
        aligned_whisper_transcriptions: The transcription text aligned with Whisper's output (small syntactic formulation differences).
        """,
    )
    parser.add_argument(
        "--dataset_train",
        type=str,
        choices=["tinyStress-15K"], # add other datasets as needed
        default="tinyStress-15K",
        help="Name of the dataset to be used for training and validation",
    )
    parser.add_argument(
        "--dataset_eval",
        type=str,
        choices=["tinyStress-15K"], # add other datasets as needed
        default="tinyStress-15K",
        help="Name of the dataset to be used for evaluation",
    )
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True,
        help="Whether to train the model (True) or only evaluate it (False)",
    )
    
    args = parser.parse_args()

    # Create training_results directory if it does not exist
    if not args.output_path:
        print("No output path provided, creating a new directory named 'training_results' in the current directory.")
        output_path = CURRENT_DIR / "training_results"
        output_path.mkdir(parents=True, exist_ok=True)
        args.output_path = str(output_path)

    if args.is_train:
        assert args.model_path is None, "If training, a model path (of the final model) must not be provided"
    else:
        assert args.model_path is not None, "If not training, a model (of the final model) path must be provided"
    
    train_or_evaluate(args)