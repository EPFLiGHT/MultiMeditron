from multimeditron.model.modalities import BaseModalityProcessor
from multimeditron.model.constants import MODALITIES_KEY, MODALITY_TYPE_KEY
from multimeditron.model.model import ChatTemplate
from multimeditron.model.prompt_tokenizers import PromptTokenizer
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase


class SamplePreprocessor:
    """
    A class designed to preprocess input data samples for multimodal models.

    This class is responsible for handling the tokenization of input samples
    and processing modality-specific data into tensors. It serves as the
    intermediary between the raw input data and the model-ready format,
    leveraging modality-specific processing logic and tokenizers.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_type: str,
        modality_processors: Dict[str, BaseModalityProcessor],
        attachment_token_idx: int,
    ):
        """
        Initialize the SamplePreprocessor.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer instance to use for tokenization.
            tokenizer_type (str): The type of tokenizer, used to look up the appropriate tokenizer class.
            modality_processors (Dict[str, BaseModalityProcessor]): A dictionary mapping modality types
                (e.g., 'image', 'audio', ...) to their respective processing classes.
            attachment_token_idx (int): The index of the attachment token used during tokenization.
        """
        self.modalities_num_embeddings = None
        self.prompt_tokenizer = PromptTokenizer(
            tokenizer=tokenizer,
            chat_template=ChatTemplate.from_name(tokenizer_type),
            modalities_num_embeddings=self.modalities_num_embeddings,
            attachment_token_idx=attachment_token_idx,
        )
        self.modality_processors = modality_processors

    def tokenize(self, samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Tokenize a batch of input samples using the prompt tokenizer.

        Args:
            samples (List[Dict[str, Any]]): A batch of input samples where each sample is a dictionary containing
                raw data to be tokenized.
            kwargs: Additional arguments to customize the tokenization process.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing tokenized samples.

        Example:

        .. code-block:: python

            samples = [
                {"conversations": 
                    {
                        "role": "user",
                        "content": "Describe the image in detail."
                    }
                },
            ]
            tokenized_samples = sample_preprocessor.tokenize(samples)
        """

        processed_samples = self.prompt_tokenizer.tokenize_samples(samples, **kwargs)
        
        return processed_samples

    def process_modality_to_tensor(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process modality-specific data in the input samples into tensors.

        Args:
            samples (List[Dict[str, Any]]): A list of input samples where each sample contains modality-specific data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary has processed modality data represented as tensors.

        Example:
    
        .. code-block:: python

            samples = [
                {"modalities": [
                    {"type": "image", "value": "image"}
                ]}
            ]
            processed_samples = sample_preprocessor.process_modality_to_tensor(samples)
        """
        processed_samples = []
        for sample in samples:
            processed_sample = sample.copy()
            processed_sample[MODALITIES_KEY] = []

            for modality in sample.get(MODALITIES_KEY, []):
                processed_sample[MODALITIES_KEY].append(
                    self.modality_processors[modality[MODALITY_TYPE_KEY]].process(modality)
                )
            processed_samples.append(processed_sample)

        return processed_samples
