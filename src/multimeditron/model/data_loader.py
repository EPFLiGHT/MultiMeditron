from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Any, Optional, Union, Iterator
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass
# from multimeditron.model.modality import ModalityWithProjection
from multimeditron.model.modalities import BaseModalityProcessor
from multimeditron.dataset.preprocessor.modality_preprocessor import SamplePreprocessor
from multimeditron.utils import pydantic_enum
from enum import IntEnum, auto
import torch
from multimeditron.model.prompt_tokenizers import MODALITIES_KEY, NUM_EMBEDDINGS_KEY
from torch.nn.utils.rnn import pad_sequence

IGNORE_TOKEN_INDEX = -100  # This value is hardcoded in the transformers library

@dataclass
class DataCollatorForMultimodal(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    modality_processors: Dict[str, BaseModalityProcessor]
    attachment_token_idx: int
    tokenizer_type: str
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    add_generation_prompt: bool = False
    max_length: Optional[int] = None

    def torch_call(self, raw_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
            Args:
            
            features (List[Dict[str, Any]]): List of batches, each Dictionary contains the following keys: 
                - input_ids (List[int]): List of input token ids.
                - labels (List[int]): List of label token ids.
                - modalities (List[Dict[str, Any]]): List of modalities, each Dictionary contains the following keys:
                    - type (str): Modality type.
                    - value (Any): Modality value.
                Each element in the list is a sample.
        """
        # Separate features by modality
        batch = {}

        text_features = {
            'input_ids' : [],
            'labels' : [],
            'attention_mask' : [],
            'modalities' : []
        }

        stackable_features = {"input_ids", "labels", "attention_mask"}

        modality_preprocessor = SamplePreprocessor(
            tokenizer=self.tokenizer,
            tokenizer_type=self.tokenizer_type,
            modality_processors=self.modality_processors,
            attachment_token_idx=self.attachment_token_idx,
        )
        
        processed_samples = modality_preprocessor.process_modality_to_tensor(raw_features)
        features = modality_preprocessor.tokenize(processed_samples, add_generation_prompt=self.add_generation_prompt)

        for sample in features:
            for name in text_features.keys():
                text_features[name].append(sample[name])
        
        # Convert list of tensors to tensor
        for key in text_features.keys():
            if key in stackable_features:
                text_features[key] = torch.stack(text_features[key])

        batch.update(text_features)

        # Create modality stacks and compute batch indices/token ranges
        modality_types = set(pm['type'] for sample in features for pm in sample["modalities"])
        multimodal_multi_idx = {modality_type: [] for modality_type in modality_types}
        multimodal_stacks = {modality_type: [] for modality_type in modality_types}

        for batch_idx, sample in enumerate(features):
            for sample_idx, pm in enumerate(sample["modalities"]):
                multimodal_multi_idx[pm['type']].append((batch_idx, pm['token_range']))
                multimodal_stacks[pm['type']].append(pm["value"])

        multimodal_batch_idx = {}
        multimodal_token_range = {}

        for modality_type in multimodal_multi_idx:
            batch_idx, token_range = zip(*multimodal_multi_idx[modality_type])
            batch_idx_exp = torch.tensor(batch_idx).repeat_interleave(torch.tensor([tr[1]-tr[0] for tr in token_range]))
            token_range_exp = torch.cat([torch.tensor(range(tr[0], tr[1])) for tr in token_range])
            multimodal_batch_idx[modality_type] = batch_idx_exp
            multimodal_token_range[modality_type] = token_range_exp

        multimodal_stacked = {}
    
        for modality_type, stack in multimodal_stacks.items():
            multimodal_stacked[modality_type] = stack


        batch['processed_multimodal_inputs'] = {
            'batch_idx': multimodal_batch_idx,
            'token_range': multimodal_token_range,
            'stacked': multimodal_stacked
        }

        # Process position ids
        attention_mask = batch["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        batch["position_ids"] = position_ids

        return batch

    def tf_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "TensorFlow is not supported for multimodal data collation.")

    def numpy_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "NumPy is not supported for multimodal data collation.")
