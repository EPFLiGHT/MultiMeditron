import abc
from transformers import PreTrainedTokenizerBase
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import copy
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from multimeditron.model.constants import (
    NUM_EMBEDDINGS_KEY, CONVERSATIONS_KEY, TEXT_KEY,
    MODALITIES_KEY, IGNORE_TOKEN_INDEX
)

class PromptTokenizer(abc.ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 attachment_token_idx: int,
                 modalities_num_embeddings: Dict[str, Optional[int]],
                 ignore_index: int = -100):
        """
        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            modalities_num_embeddings (Dict[str, int]): A dictionary mapping modality names to the number of embeddings they have.
            ignore_index (int, optional): The index to ignore. Defaults to -100.
            attachment_token (str, optional): The token to use for attachment. Defaults to "<|attachment|>".
        """
        
        self.modalities_num_embeddings = modalities_num_embeddings
        self.tokenizer = copy.deepcopy(tokenizer)
        self.ignore_index = ignore_index
        self.attachment_token_idx = attachment_token_idx
        self.pad_token_idx = self.convert_tokens_to_ids(self.tokenizer.pad_token)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize_samples(self, samples: Union[List[Dict[str, Any]], Dict[str, Any]],
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Tokenizes a sample, which can either be a text or a conversation.
        Args:
            sample (Dict[str, Any]): The sample to tokenize. It must contain either 'text' or 'conversations'.
        Returns:
            Dict[str, Any]: The tokenized sample containing 'input_ids', 'attention_mask', and 'labels'.
        """
        if isinstance(samples, dict):
            samples = [samples]

        text_samples = []
        text_modalities = []
        conversations_samples = []
        conversations_modalities = []

        for sample in samples:
            if TEXT_KEY in sample:
                text_samples.append(sample[TEXT_KEY])
                text_modalities.append(sample[MODALITIES_KEY])
            elif CONVERSATIONS_KEY in sample:
                conversations_samples.append(sample[CONVERSATIONS_KEY])
                conversations_modalities.append(sample[MODALITIES_KEY])
            else:
                raise ValueError(
                    "Each sample must contain either 'text' or 'conversations'.")
        
        tokenized_conversations = []
        if len(conversations_samples) > 0:
            tokenized_conversations = self.tokenize_conversation(conversations_samples, conversations_modalities, **kwargs)

        tokenized_texts = []
        if len(text_samples) > 0:
            tokenized_texts = self.tokenize_text(text_samples, text_modalities)
        
        tokenized = tokenized_conversations + tokenized_texts

        padded_tokenized = self.pad_tokenized(tokenized)
        return self.update_with_token_range(padded_tokenized, samples)


    def update_with_token_range(self, tokenized: Dict[str, torch.Tensor], 
                                samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            token_range = self.compute_token_range(tokenized["input_ids"][i], sample["modalities"])

            for modality, tr in zip(sample[MODALITIES_KEY], token_range):
                modality["token_range"] = tr

            processed_samples.append({
                "input_ids" : tokenized["input_ids"][i],
                "attention_mask" : tokenized["attention_mask"][i],
                "labels" : tokenized["labels"][i],
                MODALITIES_KEY : sample[MODALITIES_KEY]
            })

        return processed_samples

    def pad_tokenized(self, tokenized: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        padded_tokenized = {
                "input_ids" : [],
                "attention_mask" : [],
                "labels" : [],
        }

        # The pad token that we use for each attributes
        padding_value = {
                "input_ids" : self.pad_token_idx,
                "attention_mask" : 0,
                "labels": IGNORE_TOKEN_INDEX
        }

        # Convert list of dict into dict of list
        for sample in tokenized:
            for key in padded_tokenized.keys():
                padded_tokenized[key].append(sample[key])

        res = dict()
        for key in padded_tokenized:
            res[key] = pad_sequence(padded_tokenized[key],
                                    padding_side=self.tokenizer.padding_side, 
                                    padding_value=padding_value[key],
                                    batch_first=True)

        return res


    def tokenize_conversation(self, prompt: List[Dict[str, str]], modalities: List[List[Dict[str, Any]]], 
                              add_eos_token=True, add_generation_prompt=False) -> List[Dict[str, Any]]:
        res = self._tokenize_conversation(prompt, modalities, add_eos_token=add_eos_token, add_generation_prompt=add_generation_prompt)
        self.validate_tokenized_results(res)
        return res
    
    def tokenize_text(self, prompt: List[str], modalities: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        res = self._tokenize_text(prompt, modalities)
        self.validate_tokenized_results(res)
        return res
        
    def validate_tokenized_results(self, results: List[Dict[str, Any]]):
        for res in results:
            if not ("input_ids" in res and "attention_mask" in res and "labels" in res):
                raise ValueError(
                    "Result of tokenize_conversation must contain keys: input_ids, attention_mask and labels")

    def convert_tokens_to_ids(self, tokens: List[str]) -> int:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_num_embeddings(self, modality: Dict[str, Any]) -> int:
        if NUM_EMBEDDINGS_KEY in modality:
            return int(modality[NUM_EMBEDDINGS_KEY])

        if modality["type"] in self.modalities_num_embeddings and \
                self.modalities_num_embeddings[modality["type"]] is not None:
            return self.modalities_num_embeddings[modality["type"]]

        raise ValueError(f"Modality should contain a {NUM_EMBEDDINGS_KEY} key or \
                         you should give a num_embeddings for {modality['type']} to this PromptTokenizer")

    def compute_token_range(self, sequence_input_ids: torch.Tensor, sequence_modalities: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Compute token range for a sample

        Returns:
            A list of tuples containing elements of the form (start, end) corresponding to the start of modality in the sequence
            and end of modality in the sequence
        """
        if len(sequence_modalities) == 0:
            return []
        
        if type(sequence_input_ids) is not torch.Tensor:
            sequence_input_ids = torch.tensor(sequence_input_ids)

        modalities_indices = torch.argwhere(sequence_input_ids == self.attachment_token_idx).flatten()
        modalities_lengths = [self.get_num_embeddings(m) for m in sequence_modalities]
        
        modalities_token_start = modalities_indices[np.cumsum([0] + modalities_lengths[:-1])].tolist()
        modalities_token_range = [(start, start + length) for start, length in zip(modalities_token_start, modalities_lengths)]

        return modalities_token_range


    def expand_attachment_input_tokens(self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
                                       modalities_for_message: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expands attachment tokens in the token sequence based on the number of embeddings for each modality.

        Args:
            token_ids (torch.Tensor): The original sequence of token IDs.
            attention_mask (torch.Tensor): The attention mask corresponding to the token_ids.
            modalities_for_message (List[Dict[str, Any]]): A list of modality dictionaries, each containing modality information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The expanded token IDs and corresponding attention mask.
        """
        if len(modalities_for_message) == 0:
            return token_ids, attention_mask

        modalities_indices = torch.argwhere(
            token_ids == self.attachment_token_idx).flatten()
        modalities_names = list(
            map(lambda x: x["type"], modalities_for_message))
        
        assert len(modalities_names) == len(modalities_indices)
        assert len(attention_mask) == len(token_ids)

        # First, take all the text until the first modality (excluded)
        expanded_token_ids = [token_ids[: modalities_indices[0] - 1]]
        expanded_attention_mask = [attention_mask[: modalities_indices[0] - 1]]

        # Add the first modality
        num_embeddings = self.get_num_embeddings(modalities_for_message[0])
        expanded_token_ids.append(torch.tensor(
            [self.attachment_token_idx] * num_embeddings))
        expanded_attention_mask.append(torch.tensor([True] * num_embeddings))

        for previous_mod_idx, current_mod_idx, mod in zip(modalities_indices, modalities_indices[1:], modalities_for_message[1:]):
            # Add the text
            expanded_token_ids.append(
                token_ids[previous_mod_idx + 1: current_mod_idx])
            expanded_attention_mask.append(
                attention_mask[previous_mod_idx + 1: current_mod_idx])

            # Add the correct number of attachment tokens for the current modality
            num_embeddings = self.get_num_embeddings(mod)
            expanded_token_ids.append(torch.tensor(
                [self.attachment_token_idx] * num_embeddings))
            # Don't want to mask the attachment
            expanded_attention_mask.append(
                torch.tensor([True] * num_embeddings))

        # Add final piece of text
        last_mod_idx = modalities_indices[-1]
        expanded_token_ids.append(token_ids[last_mod_idx + 1:])
        expanded_attention_mask.append(attention_mask[last_mod_idx + 1:])

        expanded_token_ids = torch.cat(expanded_token_ids)
        expanded_attention_mask = torch.cat(expanded_attention_mask)

        return expanded_token_ids, expanded_attention_mask

    
    def _tokenize_text(self, text: List[str], modalities: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(text, str):
            text = [text]

        outputs = self.tokenizer(text, return_tensors="pt")

        tokenized_results = []
        
        # Iterate on each sample of the batch
        for i in range(len(text)):
            # Expand attachment tokens
            input_ids, attention_mask = self.expand_attachment_input_tokens(
                token_ids=outputs["input_ids"][i],
                attention_mask=outputs["attention_mask"][i],
                modalities_for_message=modalities[i]
            )
        
            # We don't want to predict image tokens
            labels = torch.where(input_ids == self.attachment_token_idx, self.ignore_index, input_ids)
            
            # Don't want to predict the pad tokens
            labels = torch.where(attention_mask == 0, IGNORE_TOKEN_INDEX, input_ids)

            tokenized_results.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        return tokenized_results


class Llama3PromptTokenizer(PromptTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 modalities_num_embeddings: Dict[str, Optional[int]],
                 attachment_token_idx: int,
                 ignore_index: int = -100):
        super().__init__(
            tokenizer=tokenizer,
            modalities_num_embeddings=modalities_num_embeddings,
            ignore_index=ignore_index,
            attachment_token_idx=attachment_token_idx,
        )
    
    def _tokenize_conversation(self, conversation: List[List[Dict[str, str]]],
                           modalities: List[List[Dict[str, Any]]], 
                           add_eos_token=True, 
                           add_generation_prompt=False) -> List[Dict[str, Any]]:
        # Expand the attachments
        tokenized_results = []
        for conv, mod in zip(conversation, modalities):
            outputs = self.tokenizer.apply_chat_template(
                    conv, add_eos_token=add_eos_token,
                    return_dict=True, return_tensors="pt", 
                    add_generation_prompt=add_generation_prompt,
            )

            input_ids, attention_mask = self.expand_attachment_input_tokens(
                token_ids=outputs["input_ids"].flatten(),
                attention_mask=outputs["attention_mask"].flatten(),
                modalities_for_message=mod
            )

            # Don't want to predict pad tokens
            labels = torch.where(attention_mask == 0, IGNORE_TOKEN_INDEX, input_ids)

            left_tag = self.tokenizer.encode("<|start_header_id|>system<|end_header_id|>", add_special_tokens=False)
            right_tag = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            labels = replace_between_tags_v2(labels, left_tag=left_tag, right_tag=right_tag)

            left_tag = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>", add_special_tokens=False)
            labels = replace_between_tags_v2(labels, left_tag=left_tag, right_tag=right_tag)

            tokenized_results.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        return tokenized_results

class ApertusPromptTokenizer(PromptTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 modalities_num_embeddings: Dict[str, Optional[int]],
                 attachment_token_idx: int,
                 ignore_index: int = -100):
        super().__init__(
            tokenizer=tokenizer,
            modalities_num_embeddings=modalities_num_embeddings,
            ignore_index=ignore_index,
            attachment_token_idx=attachment_token_idx,
        )
        
    def _tokenize_conversation(self, conversation: List[List[Dict[str, str]]],
                               modalities: List[List[Dict[str, Any]]], 
                               add_eos_token=True, 
                               add_generation_prompt=False,
                               padding=True) -> List[Dict[str, Any]]:

        tokenized_results = []

        # Expand the attachments
        for conv, mod in zip(conversation, modalities):
            outputs = self.tokenizer.apply_chat_template(
                    conv, add_eos_token=add_eos_token,
                    return_dict=True, return_tensors="pt", 
                    add_generation_prompt=add_generation_prompt,
            )

            input_ids, attention_mask = self.expand_attachment_input_tokens(
                token_ids=outputs["input_ids"].flatten(),
                attention_mask=outputs["attention_mask"].flatten(),
                modalities_for_message=mod
            )
            
            # Don't want to predict pad tokens
            labels = torch.where(attention_mask == 0, IGNORE_TOKEN_INDEX, input_ids)

            # Apertus-specific: mask out system messages
            # <SPECIAL_61>system_content<SPECIAL_62> (tokens 61 and 62)
            system_start = self.tokenizer.encode("<SPECIAL_61>", add_special_tokens=False)
            system_end = self.tokenizer.encode("<SPECIAL_62>", add_special_tokens=False)
            labels = replace_between_tags_v2(labels, left_tag=system_start, right_tag=system_end)

            # Apertus-specific: mask out developer messages
            # <SPECIAL_63>developer_content<SPECIAL_64> (tokens 63 and 64)
            developer_start = self.tokenizer.encode("<SPECIAL_63>", add_special_tokens=False)
            developer_end = self.tokenizer.encode("<SPECIAL_64>", add_special_tokens=False)
            labels = replace_between_tags_v2(labels, left_tag=developer_start, right_tag=developer_end)

            # Apertus-specific: mask out user messages
            # <SPECIAL_65>user_content<SPECIAL_66> (tokens 65 and 66)
            user_start = self.tokenizer.encode("<SPECIAL_65>", add_special_tokens=False)
            user_end = self.tokenizer.encode("<SPECIAL_66>", add_special_tokens=False)
            labels = replace_between_tags_v2(labels, left_tag=user_start, right_tag=user_end)

            tokenized_results.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        return tokenized_results


def find_tag_pos(tensor, tag):
    tag = torch.tensor(tag)
    windows = tensor.unfold(0, tag.size(0), 1)
    matches = (windows == tag).all(dim=1)
    return torch.nonzero(matches).squeeze(1)

def replace_between_tags_v2(tensor, left_tag, right_tag, replace_value=-100):
    start_positions = find_tag_pos(tensor, left_tag)
    end_positions = find_tag_pos(tensor, right_tag)

    end_positions = end_positions[
        torch.searchsorted(end_positions, start_positions)]

    for start, end in zip(start_positions, end_positions):
        tensor[start:end + len(right_tag)] = replace_value
    
    return tensor


TOKENIZER_MAP = {
    "llama": Llama3PromptTokenizer,
    "apertus": ApertusPromptTokenizer
}

