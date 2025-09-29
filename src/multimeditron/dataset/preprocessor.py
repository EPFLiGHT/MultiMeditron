from multimeditron.model.modalities import BaseModalityProcessor
from multimeditron.model.prompt_tokenizers import MODALITIES_KEY, TOKENIZER_MAP
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase


class SamplePreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_type: str,
        modality_processors: Dict[str, BaseModalityProcessor],
        attachment_token_idx: int,
    ):
        self.modalities_num_embeddings = None
        if modality_processors is not None:
            self.modalities_num_embeddings = {
                mod_name: processor.num_patches_per_entry for mod_name, processor in modality_processors.items()
            }
        
        tokenizer_cls = TOKENIZER_MAP[tokenizer_type]

        self.prompt_tokenizer = tokenizer_cls(
            tokenizer=tokenizer,
            modalities_num_embeddings=self.modalities_num_embeddings,
            attachment_token_idx=attachment_token_idx,
        )
        self.modality_processors = modality_processors

    def tokenize(self, samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        # Process text input
        # tokenized_results is a List of Dict (batch of tokenized samples)
        processed_samples = self.prompt_tokenizer.tokenize_samples(samples, **kwargs)
        
        return processed_samples

    def process_modality_to_tensor(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_samples = []
        for sample in samples:
            processed_sample = sample.copy()
            processed_sample[MODALITIES_KEY] = []

            for modality in sample[MODALITIES_KEY]:
                processed_sample[MODALITIES_KEY].append(
                    self.modality_processors[modality["type"]].process(modality)
                )
            processed_samples.append(processed_sample)

        return processed_samples