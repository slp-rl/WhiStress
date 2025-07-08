import torch
import re
from datasets import load_dataset
from scipy.signal import resample

class DSProcessor:
    """
    Dataset processor for speech emphasis detection tasks.
    
    Handles dataset loading, audio processing, tokenization, and preparing
    training examples with emphasis annotations.
    
    Attributes:
        processor: Whisper processor for audio and text processing
        ds_name: HuggingFace dataset name
        hyperparameters: Dictionary of configuration parameters
        hf_token: HuggingFace API token for accessing datasets
    """
    def __init__(
        self,
        processor,
        hyperparameters,
        ds_name,
        hf_token
    ):
        self.processor = processor
        self.ds_name = ds_name
        self.hyperparameters = hyperparameters
        self.hf_token = hf_token

    def load_intonation_dataset(self):
        """
        Load a dataset from the Hugging Face Hub.
        
        Returns:
            Dataset object loaded from Hugging Face Hub
        """
        # Load our dataset from the Hugging Face Hub
        intonation_dataset = load_dataset(self.ds_name, trust_remote_code=True, token=self.hf_token)
        return intonation_dataset

    def map_words_to_tokens(self, example, transcription_column_name):
        """
        Maps words in the transcription to their corresponding token IDs.
        
        Creates a dictionary mapping where keys are words (only characters from 
        the alphabet) and values are lists of token IDs representing those words
        in the tokenized transcription.
        
        Args:
            example: Dataset example containing transcription text
            transcription_column_name: Name of the column containing transcription text
            
        Returns:
            Example with added map_dict_{transcription_column_name} field containing:
            - 'keys': List of word strings
            - 'values': List of token ID lists for each word
        """
        # This function returns a dictionary where keys (words only from alphabet) are mapped to token IDS (values) in the transcription.
        # e.g. map_dict = {'Kitty': [42, 9760], 'smiled': [13541], 'and': [290], 'replied': [8712], 'Thank': [10449], 'you': [345], 'Spot': [15899]} ->
        # returned value:
        # keys: ['Kitty', 'smiled', 'and', 'replied', 'Thank', 'you', 'Spot']
        # values: [[42, 9760], [13541], [290], [8712], [10449], [345], [15899]]
        def contains_no_alpha(s):
            return not re.search(r"[a-zA-Z\']", s)

        def remove_non_alpha(s):
            return re.sub(r"[^a-zA-Z\']", "", s)

        tokens = self.processor.tokenizer.tokenize(example[transcription_column_name])
        tokens_ids = self.processor.tokenizer.convert_tokens_to_ids(tokens)

        map_dict = {}
        current_word = tokens[0]
        current_words_tokens = [tokens_ids[0]]
        dict_elem = 0

        for token_ids, token in zip(tokens_ids[1:], tokens[1:]):
            # Whisper uses 'Ġ' to denote the start of a new word
            if token.startswith("Ġ"):
                if current_word:
                    map_dict[f"{remove_non_alpha(current_word)} {dict_elem}"] = (
                        current_words_tokens
                    )
                    current_word = ""
                    current_words_tokens = []
                    dict_elem += 1
                # start a new word (remove the leading 'Ġ')
                current_word = token[1:]
            else:
                # continue the current word
                current_word += token
            # if we came across a token that contains no alphabet characters, we skip it, except for commas
            if not contains_no_alpha(token):
                current_words_tokens.append(token_ids)

        # Add the last word
        if current_word:
            map_dict[f"{remove_non_alpha(current_word)} {dict_elem}"] = (
                current_words_tokens
            )

        correct_map_dict = {
            f"map_dict_{transcription_column_name}": {
                "keys": [str(key).split(" ")[0] for key in map_dict.keys()],
                "values": map_dict.values(),
            }
        }
        example.update(correct_map_dict)
        return example

    def emphasized_tokens(self, example, 
                          transcription_column_name,
                          emphasis_indices_column_name="emphasis_indices"):
        """
        Creates a binary vector marking which tokens are emphasized in the transcription.
        
        Handles two different formats of emphasis annotations:
        1. List of indices indicating emphasized word positions
        2. Binary vector with 1s for emphasized words and 0s otherwise
        
        Args:
            example: Dataset example containing transcription and emphasis indices
            transcription_column_name: Name of the column containing transcription text
            emphasis_indices_column_name: Name of the column containing emphasis annotations
            
        Returns:
            Example with added labels_head_{transcription_column_name} field containing 
            a binary tensor with 1s for emphasized tokens and 0s otherwise
        """
        # This function returns a binary vector with 1 entries for emphasized tokens in the transcription (including special tokens with 0).
        curr_tokenized_sentence = self.processor.tokenizer(
            example[transcription_column_name]
        ).input_ids
        curr_values = example[f"map_dict_{transcription_column_name}"]["values"]
        # if not len(curr_values) == len(example[emphasis_indices_column_name]):
        if emphasis_indices_column_name=="emphasis_indices" or emphasis_indices_column_name=="emphasis_indices_nru":
            # if we reached here, then it means the emphsasis indices were passed as a list of indices of the 
            # location where emphsized words appear
            concatenated_values = []
            if len(curr_values) > example[emphasis_indices_column_name][-1]:
                concatenated_values = [
                    item for elem in example[emphasis_indices_column_name] for item in curr_values[elem]
                ]
            else:
                print(example['sentence_index'])                
        else:
            # if we reached here, then it means the emphsasis indices were passed as a binary vecotr
            # in the length of the number of words where 1 symbolizes emphsis and 0 otherwise
            indices = [index for index, value in enumerate(example[emphasis_indices_column_name]) if value == 1]
            concatenated_values = [
                item for elem in indices for item in curr_values[elem]
            ]
        j = 0
        emphasized_words = []
        for token in curr_tokenized_sentence:
            binary = 0
            if j < len(concatenated_values):
                if token == concatenated_values[j]:
                    binary = 1
                    j += 1
            emphasized_words.append(binary)
        example[f"labels_head_{transcription_column_name}"] = torch.tensor(emphasized_words)
        return example

    def prepare_dataset(self, example, transcription_column_name):
        """
        Prepares the final dataset example by adding token labels.
        
        Tokenizes the transcription text and adds the resulting token IDs as labels.
        Verifies that the emphasis head labels match the length of token labels.
        
        Args:
            example: Dataset example containing the transcription
            transcription_column_name: Name of the column containing transcription text
            
        Returns:
            Example with added labels_{transcription_column_name} field containing 
            tokenized transcription IDs
        """
        example[f"labels_{transcription_column_name}"] = self.processor.tokenizer(example[transcription_column_name]).input_ids
        assert len(example[f"labels_head_{transcription_column_name}"]) == len(example[f"labels_{transcription_column_name}"])
        return example
    
    def aligned_whisper_transcriptions(self, example, model):
        """
        Generates Whisper model transcriptions and checks alignment with ground truth.
        
        Resamples audio to 16kHz, processes it through Whisper, and compares the 
        generated transcription with the ground truth. If word counts match, keeps
        the Whisper transcription; otherwise, marks as misaligned.
        
        Args:
            example: Dataset example containing audio and transcription
            model: Whisper model for generating transcriptions
            
        Returns:
            Example with added input_features and aligned_whisper_transcriptions fields
        """
        num_samples = int(len(example["audio"]["array"]) * 16000 / example['audio']['sampling_rate'])
        downsampled_audio_array = resample(example["audio"]["array"], num_samples)

        example["input_features"] = self.processor(
            downsampled_audio_array,
            return_tensors="pt",
            sampling_rate=16000,
            truncation=True,
        ).input_features
        # encode target text to label ids
        token_ids = model.whisper_model.generate(input_features=example['input_features'].to('cuda'))
        transcription_whisper = self.processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        example['aligned_whisper_transcriptions'] = ''
        transcription_whisper_clean = re.sub(r"[^a-zA-Z\'\- ]", "", transcription_whisper)
        transcription_gt_clean = re.sub(r"[^a-zA-Z\'\- ]", "", example['transcription'])
        if len(transcription_whisper_clean.lstrip().lower().split(" ")) == len(transcription_gt_clean.lstrip().lower().split(" ")):
            example['aligned_whisper_transcriptions'] = transcription_whisper
        else:
            print(f"example text: \"{example['transcription']}\" doesn't match whisper's text: \"{transcription_whisper}\"")
        return example
    
    def filter_misaligned_samples(self, example, transcription_column_name):
        """
        Filters out examples where Whisper transcription doesn't align with ground truth.
        
        Args:
            example: Dataset example containing aligned_whisper_transcriptions
            transcription_column_name: Name of the column to check for emptiness
            
        Returns:
            Boolean indicating whether the example should be kept (True) or filtered out (False)
        """
        return example[transcription_column_name] != ''

    def filter_incorrect_transcription(self, example, transcription_column_name):
        """
        Filters out examples where the token mapping doesn't match word count.
        
        Checks if the number of words in the transcription matches the number of
        non-empty token lists in the word-to-token mapping.
        
        Args:
            example: Dataset example containing transcription and token mapping
            transcription_column_name: Name of the column containing transcription text
            
        Returns:
            Boolean indicating whether the example should be kept (True) or filtered out (False)
        """
        transcription = example[transcription_column_name]
        values = [val for val in example[f"map_dict_{transcription_column_name}"]["values"] if len(val) != 0]
        return len(transcription.split(" ")) == len(values)

    def preprocess(self, 
                   model,
                   transcription_column_name,
                   emphasis_indices_column_name="emphasis_indices"):
        """
        Full preprocessing pipeline for the dataset.
        
        Applies a sequence of preprocessing steps:
        1. Generates Whisper transcriptions and checks alignment
        2. (Optional) Filters misaligned samples
        3. Maps words to tokens
        4. Creates binary emphasis vectors
        5. Prepares final dataset format
        
        Args:
            model: Whisper model for generating transcriptions
            transcription_column_name: Name of the column containing transcription text
            emphasis_indices_column_name: Name of the column containing emphasis annotations
            
        Returns:
            Fully preprocessed dataset ready for training
        """
        proccess_methods = {
            "aligned_whisper_transcriptions": lambda example: self.aligned_whisper_transcriptions(example, model),
            "filter_misaligned_samples": lambda example: self.filter_misaligned_samples(example, transcription_column_name),
            "map_words_to_tokens": lambda example: self.map_words_to_tokens(example, transcription_column_name=transcription_column_name),
            "filter_incorrect_transcription": lambda example: self.filter_incorrect_transcription(example, transcription_column_name=transcription_column_name),
            "emphasized_tokens": lambda example: self.emphasized_tokens(example, emphasis_indices_column_name=emphasis_indices_column_name, transcription_column_name=transcription_column_name),
            "prepare_dataset": lambda example: self.prepare_dataset(example, transcription_column_name=transcription_column_name)
        }
        intonation_dataset = self.load_intonation_dataset()
        intonation_dataset_2 = intonation_dataset.map(
            proccess_methods["aligned_whisper_transcriptions"], num_proc=1
        )
        
        # NOTE - if you wish to filter out samples with misaligned transcriptions, uncomment the following line
        # intonation_dataset_2 = intonation_dataset_2.filter(
        #     proccess_methods["filter_misaligned_samples"], num_proc=1
        # )
        
        # NOTE - diferentiate between the two types of emphasis indices: 
            # option 1 : passed as a binary vector with 1 symbolizing emphsis token and 0 otherwise
            # option 2 : passed as a list of indices of the location where emphsized words appear
        if emphasis_indices_column_name != 'emphasis_indices' and "emphasis_indices" in intonation_dataset_2.column_names:
            intonation_dataset_2 = intonation_dataset_2.rename_column("emphasis_indices", emphasis_indices_column_name)
        
        intonation_dataset_3 = intonation_dataset_2.map(
            proccess_methods["map_words_to_tokens"], num_proc=1
        )
        # need to prepare a binary emphasis vector for each sample
        intonation_dataset_4 = intonation_dataset_3.map(
            proccess_methods["emphasized_tokens"], num_proc=1
        )
        intonation_dataset_5 = intonation_dataset_4.map(
            proccess_methods["prepare_dataset"], num_proc=1
        )
        return intonation_dataset_5

    def get_train_dataset(self, 
                                model, 
                                transcription_column_name, 
                                emphasis_indices_column_name="emphasis_indices",
                                columns_to_remove=[]
                            ):
        """
        Get the final preprocessed training dataset.
        
        Runs the full preprocessing pipeline, removes unnecessary columns,
        and sets the format to PyTorch tensors.
        
        Args:
            model: Whisper model for generating transcriptions
            transcription_column_name: Name of the column containing transcription text
            emphasis_indices_column_name: Name of the column containing emphasis annotations
            columns_to_remove: List of column names to remove from the final dataset
            
        Returns:
            Preprocessed dataset in PyTorch format, ready for training
        """
        preproc_intonation_dataset = self.preprocess(
            model=model,
            emphasis_indices_column_name=emphasis_indices_column_name,
            transcription_column_name=transcription_column_name
        )
        preproc_intonation_dataset = preproc_intonation_dataset.remove_columns(columns_to_remove)
        preproc_intonation_dataset.set_format("torch")
        return preproc_intonation_dataset

