from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass
# from multimeditron.model.modality import ModalityWithProjection
from multimeditron.dataset.loader import BaseModalityLoader
from multimeditron.model.modalities import BaseModalityProcessor
from multimeditron.dataset.preprocessor import SamplePreprocessor
import torch
from multimeditron.model.constants import MODALITIES_KEY, MODALITY_TYPE_KEY, MODALITY_VALUE_KEY

IGNORE_TOKEN_INDEX = -100  # This value is hardcoded in the transformers library

@dataclass
class DataCollatorForMultimodal(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    modality_processors: Dict[str, BaseModalityProcessor]
    modality_loaders: Dict[str, BaseModalityLoader]
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

        # Load modality values
        raw_features = [BaseModalityLoader.merge_modalities(f, self.modality_loaders) for f in raw_features]

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
        modality_types = set(pm[MODALITY_TYPE_KEY] for sample in features for pm in sample[MODALITIES_KEY])
        multimodal_multi_idx = {modality_type: [] for modality_type in modality_types}
        multimodal_stacks = {modality_type: [] for modality_type in modality_types}

        for batch_idx, sample in enumerate(features):
            for pm in sample[MODALITIES_KEY]:
                multimodal_multi_idx[pm[MODALITY_TYPE_KEY]].append((batch_idx, pm['token_range']))
                multimodal_stacks[pm[MODALITY_TYPE_KEY]].append(pm[MODALITY_VALUE_KEY])

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
