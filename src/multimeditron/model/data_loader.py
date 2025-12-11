from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass
# from multimeditron.model.modality import ModalityWithProjection
from multimeditron.dataset.loader import BaseModalityLoader
from multimeditron.model.modalities import BaseModalityProcessor
from multimeditron.dataset.sample_preprocessor import SamplePreprocessor
from multimeditron.model.model import ChatTemplate
import torch
from multimeditron.model.constants import MODALITIES_KEY, MODALITY_TYPE_KEY, MODALITY_VALUE_KEY, IGNORE_TOKEN_INDEX, POSITION_IDS_KEY

@dataclass
class DataCollatorForMultimodal(DataCollatorMixin):
    """
    A data collator for multimodal datasets that prepares batches of data for input into models.

    This class is designed to handle datasets containing multiple modalities (e.g., text, images, etc.).
    It processes and collates the data into a format suitable for multimodal model training and inference.
    """

    tokenizer: PreTrainedTokenizerBase
    modality_processors: Dict[str, BaseModalityProcessor]
    modality_loaders: Dict[str, BaseModalityLoader]
    attachment_token: str
    chat_template: ChatTemplate
    add_generation_prompt: bool = False
    use_2d_position_ids: bool = False
    return_tensors: str = "pt"

    @torch.no_grad()
    def torch_call(self, raw_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of multimodal data.

        Args:
            raw_features (List[Dict[str, Any]]):
                A batch consisting of dictionaries where each dictionary represents a sample. Each sample must have:

                - conversations (List[Dict[str, str]]):
                    Conversation history with roles and content.
                - modalities (List[Dict[str, Any]]):
                    Information about additional modalities in the sample. Each modality contains:
                    - type (str): Type of the modality (e.g., 'image', 'audio').
                    - value (Any): Data associated with the modality.

                or:

                - text (str):
                    The text content of the sample.
                - modalities (List[Dict[str, Any]]):
                    Information about additional modalities in the sample. Each modality contains:

                    - type (str): Type of the modality (e.g., 'image', 'audio').
                    - value (Any): Data associated with the modality.

        Returns:
            Dict[str, Any]:
                A dictionary structured as follows:

                    - input_ids (torch.Tensor):
                        Batch tensor of tokenized input sequences.
                    - labels (torch.Tensor):
                        Batch tensor of tokenized labels.
                    - attention_mask (torch.Tensor):
                        Batch tensor indicating padded positions (0 for padding, 1 otherwise).
                    - position_ids (torch.Tensor):
                        Batch tensor of position indices for each token in the sequence.
                    - processed_multimodal_inputs (Dict[str, Any]):

                    Contains processed modality data with keys:

                        - batch_idx (Dict[str, torch.Tensor]):
                            Maps modality types to tensors indicating which batch sample each token belongs to.
                        - token_range (Dict[str, torch.Tensor]):
                            Maps modality types to tensors specifying the token range for each modality.
                        - stacked (Dict[str, List[Any]]):
                            Stores lists of modality values grouped by their types.

        The function performs the following steps:

            1. Separates input features by modality.
            2. Loads and processes modality-related data.
            3. Converts lists of modality features into tensors using the modality processors.
            4. Tokenizes text data by expanding the modality placeholders to the right amount.
            5. Computes positional and attention masks for sequence data.
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
            chat_template=self.chat_template,
            modality_processors=self.modality_processors,
            attachment_token=self.attachment_token,
        )

        # Load modality values
        raw_features = [BaseModalityLoader.load_modalities(f, self.modality_loaders) for f in raw_features]

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

        if self.use_2d_position_ids:
            # Create 2D position ids (batch_size, seq_length, 2)
            position_ids = position_ids.unsqueeze(-1).repeat(1, 1, 2)

            # Iterate over each modality and optionally override position_ids if required.
            for batch_idx, sample in enumerate(features):
                for pm in sample[MODALITIES_KEY]:
                    # If the POSITION_IDS_KEY is present, it indicates that the modality processor
                    if POSITION_IDS_KEY in pm:
                        token_range = pm['token_range']
                        modality_position_ids = pm[POSITION_IDS_KEY]

                        # Ensure the shape of modality_position_ids is correct
                        # Expected shape: (token_range[1] - token_range[0], 2)
                        if modality_position_ids.shape[0] != (token_range[1] - token_range[0]) or modality_position_ids.shape[1] != 2 or \
                            len(modality_position_ids.shape) != 2:
                            raise ValueError(
                                f"Modality processor for {pm[MODALITY_TYPE_KEY]} returned position_ids with incorrect shape. "
                                f"Expected ({token_range[1] - token_range[0]}, 2), got {modality_position_ids.shape}."
                            )
                        
                        old_last_position_ids = position_ids[batch_idx, token_range[1] - 1, :].clone() if token_range[0] > 0 else torch.tensor([0, 0]).long()
                        modality_position_ids += position_ids[batch_idx, token_range[0], :].unsqueeze(0) # [1, 2] because of broadcasting
                        next_last_position_ids = modality_position_ids[-1, :].max().unsqueeze(0).expand(2) # [2]

                        position_ids[batch_idx, token_range[0]:token_range[1], :] = modality_position_ids

                        # Because images that are 2D embedded use fewer position ids than if they were 1D embedded, we need to shift the subsequent
                        # position ids accordingly.
                        position_ids[batch_idx, token_range[1]:, :] += (next_last_position_ids - old_last_position_ids).unsqueeze(0) # [1, 2] because of broadcasting
        else:
            if any(any(POSITION_IDS_KEY in pm for pm in sample[MODALITIES_KEY]) for sample in features):
                print(
                    "Warning: Some modality processors have specified a position_ids, currently unsupported by the collator."
                    "Currently the collator only supports 2D (or 1D position_ids), if you want a different behavior please implement your own collator,"
                    "or modify the model to accept custom position_ids per modality."
                )

        batch["position_ids"] = position_ids

        return batch

    def tf_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Placeholder for TensorFlow integration.

        This function raises a NotImplementedError indicating that TensorFlow support is not implemented.

        Args:
            features (List[Dict[str, Any]]):
                A batch consisting of dictionaries where each dictionary represents a sample.

        Raises:
            NotImplementedError: Always raised to indicate that TensorFlow support is not available.

        Alternatives:
            Users can consider implementing a TensorFlow-specific collator if required for their use case.
        """
        raise NotImplementedError(
            "TensorFlow is not supported for multimodal data collation.")

    def numpy_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Placeholder for NumPy integration.

        This function raises a NotImplementedError indicating that NumPy support is not implemented.

        Args:
            features (List[Dict[str, Any]]):
                A batch consisting of dictionaries where each dictionary represents a sample.

        Raises:
            NotImplementedError: Always raised to indicate that NumPy support is not available.

        Alternatives:
            Users can consider implementing a NumPy-specific collator if required for their use case.
        """
        raise NotImplementedError(
            "NumPy is not supported for multimodal data collation.")
