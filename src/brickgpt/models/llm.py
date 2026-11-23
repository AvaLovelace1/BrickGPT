import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def get_default_device() -> str:
    """
    Returns "cuda" if available, otherwise "xpu" (Intel GPU) if available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


class LLM:
    """
    A small wrapper class for a language model.
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device if device is not None else get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.kv_cache = None
        self.kv_cache_saved = None
        self.input_ids_cache = None
        self.input_ids_cache_saved = None

    def __call__(
            self,
            prompt: str | torch.Tensor | None = None,
            return_as_ids: bool = False,
            return_dict: bool = False,
            **kwargs,
    ):
        """
        Generates text, given a prompt.
        """

        # If prompt is None, continue generation from previously generated tokens
        if prompt is None:
            prompt = self.input_ids_cache
        else:
            self.reset_cache()

        # If prompt is a string, encode it into token ids
        if isinstance(prompt, str):
            encoded_input = self.tokenizer(prompt, return_tensors='pt')
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
        else:
            input_ids = prompt.to(self.device)
            attention_mask = torch.ones_like(input_ids)

        # Run generation
        output_dict = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            num_return_sequences=1,
            past_key_values=self.kv_cache,
            return_dict_in_generate=True,
            **kwargs,
        )
        self.input_ids_cache = output_dict['sequences']

        # Return result as token ids or as a string
        input_length = input_ids.shape[1]
        result_ids = output_dict['sequences'][0][input_length:]
        result = result_ids if return_as_ids else self.tokenizer.decode(result_ids)

        return (result, output_dict) if return_dict else result

    def reset_cache(self) -> None:
        self.kv_cache = DynamicCache()

    def save_state(self) -> None:
        self.kv_cache_saved = copy.deepcopy(self.kv_cache)
        self.input_ids_cache_saved = self.input_ids_cache

    def rollback_to_saved_state(self) -> None:
        self.kv_cache = self.kv_cache_saved
        self.input_ids_cache = self.input_ids_cache_saved
