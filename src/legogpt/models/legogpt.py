import copy
import functools
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, LogitsProcessorList

from legogpt.data import max_brick_dimension, LegoStructure, LegoBrick
from .llm import LLM


@dataclass
class LegoGPTConfig:
    model_name_or_path: str = field(
        default='AvaLovelace/LegoGPT',
        metadata={'help': 'Model checkpoint for weights initialization.'},
    )
    world_dim: int = field(
        default=20,
        kw_only=True,
        metadata={'help': 'The dimension of the box in which the generated LEGO should fit. '
                          'Bricks outside this box are considered out of bounds.'},
    )
    max_bricks: int = field(
        default=100,
        kw_only=True,
        metadata={'help': 'The maximum number of bricks per generated LEGO structure.'},
    )
    max_brick_rejections: int = field(
        default=500,
        kw_only=True,
        metadata={'help': 'The maximum number of rejections per generated brick during rejection sampling. '
                          'Set to 0 if you want to disable rejection sampling.'},
    )
    use_logit_masking: bool = field(
        default=True,
        kw_only=True,
        metadata={'help': 'Whether to use logit masking during inference '
                          'to enforce compliance with the LEGO brick syntax. '
                          'If False, the LEGO brick will be checked for validity after generation.'},
    )
    max_regenerations: int = field(
        default=100,
        kw_only=True,
        metadata={'help': 'The maximum number of times to roll back and regenerate the LEGO structure '
                          'if it is physically unstable. '
                          'Set to 0 if you want to disable physics-informed rollback.'},
    )
    temperature: float = field(
        default=0.6,
        kw_only=True,
        metadata={'help': 'The temperature to use when sampling from the LLM.'},
    )
    temperature_increase: float = field(
        default=0.01,
        kw_only=True,
        metadata={'help': 'The amount by which to increase the temperature '
                          'after each "already_rejected" brick during rejection sampling.'},
    )
    max_temperature: float = field(
        default=2.0,
        kw_only=True,
        metadata={'help': 'The maximum temperature to increase to during rejection sampling.'},
    )
    top_k: int = field(
        default=20,
        kw_only=True,
        metadata={'help': 'The number of top tokens to sample from the LLM. '
                          'Has no effect if use_logit_masking=True.'},
    )
    top_p: float = field(
        default=1.0,
        kw_only=True,
        metadata={'help': 'The cumulative probability threshold for nucleus sampling. '
                          'Has no effect if use_logit_masking=True.'},
    )
    instruction_format: Literal['legogpt', 'few_shot', 'zero_shot'] = field(
        default='legogpt',
        kw_only=True,
        metadata={'help': 'The format of the LEGO-generating instruction to give to the LLM.'},
    )


class LegoGPT:
    def __init__(self, cfg: LegoGPTConfig):
        self.world_dim = cfg.world_dim
        self.max_bricks = cfg.max_bricks
        self.max_brick_rejections = cfg.max_brick_rejections
        self.use_logit_masking = cfg.use_logit_masking
        self.max_regenerations = cfg.max_regenerations
        self.temperature = cfg.temperature
        self.temperature_increase = cfg.temperature_increase
        self.max_temperature = cfg.max_temperature
        self.top_k = cfg.top_k
        self.top_p = cfg.top_p
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        instruction_fns = {
            'legogpt': create_instruction,
            'few_shot': create_instruction_few_shot,
            'zero_shot': create_instruction_zero_shot,
        }
        self.instruction_fn = instruction_fns[cfg.instruction_format]

        self.llm = LLM(cfg.model_name_or_path, self.device)

    def __call__(self, caption: str) -> dict:
        lego = None
        starting_lego = LegoStructure([])
        rejection_reasons = Counter()
        regeneration_num = None

        # Generate LEGO structure. If it is unstable, remove all bricks after the first unstable brick and regenerate.
        for regeneration_num in range(self.max_regenerations + 1):
            if regeneration_num > 0:
                print(f"\nRegeneration attempt {regeneration_num}/{self.max_regenerations}")
            lego, rejection_reasons_lego = self._generate_structure(caption, starting_lego=starting_lego)
            rejection_reasons.update(rejection_reasons_lego)
            if lego.is_stable():
                break
            if regeneration_num == self.max_regenerations:
                if self.max_regenerations > 0:
                    warnings.warn(f'Failed to generate a stable structure after {regeneration_num + 1} attempts.\n')
                break
            starting_lego = _remove_all_bricks_after_first_unstable_brick(lego)

        return {
            'lego': lego,
            'rejection_reasons': rejection_reasons,
            'n_regenerations': regeneration_num,
        }

    def _generate_structure(
            self,
            caption: str,
            starting_lego: LegoStructure = LegoStructure([]),
    ) -> (LegoStructure, Counter):
        """
        Generates a LEGO structure based on the given caption, starting with a partial LEGO structure.
        :param caption: A caption for the LEGO structure to be generated.
        :param starting_lego: A partial LEGO structure to which the generated bricks will be added.
        :return: A tuple containing the generated LEGO structure and a brick rejection reasons.
        """
        starting_lego = copy.deepcopy(starting_lego)

        # Construct prompt
        starting_lego_txt = starting_lego.to_txt()
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': self.instruction_fn(caption)},
        ]
        if starting_lego_txt:  # Continue generation from a partial structure
            messages.append({'role': 'assistant', 'content': starting_lego_txt})
            prompt = self.llm.tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors='pt')
        else:
            prompt = self.llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

        # Generate bricks with rejection sampling
        rejection_reasons = Counter()
        total_rejections = 0
        for brick_num in range(self.max_bricks):
            # Print progress
            progress = (brick_num / self.max_bricks) * 100
            print(f"\rGenerating brick {brick_num + 1}/{self.max_bricks} ({progress:.1f}%) - Total rejections: {total_rejections}", end="")

            brick, rejection_reasons_brick = self.generate_brick_with_rejection_sampling(
                prompt if brick_num == 0 else None, lego=starting_lego
            )
            if not brick:  # EOS token was generated
                print()  # New line after progress
                break
            rejection_reasons.update(rejection_reasons_brick)
            total_rejections += sum(rejection_reasons_brick.values())
            starting_lego.add_brick(LegoBrick.from_txt(brick))

        print()  # New line after progress
        return starting_lego, rejection_reasons

    def generate_brick_with_rejection_sampling(
            self,
            prompt: str | None = None,
            lego: LegoStructure = LegoStructure([]),
    ) -> (str, Counter):
        """
        Generates a LEGO brick to add to the LEGO structure, using rejection sampling to ensure the brick is valid.
        """
        rejection_reasons = Counter()
        rejected_bricks = set()

        brick = ''
        temperature = self.temperature
        for generation_num in range(self.max_brick_rejections + 1):
            self.llm.save_state()
            brick = self.generate_brick(prompt, temperature=temperature)
            if not brick:  # EOS token was generated
                break

            # Check if the generated brick is valid
            add_brick_result = self._try_adding_brick(brick, lego, rejected_bricks)
            if add_brick_result == 'success':
                break
            if generation_num == self.max_brick_rejections:
                if self.max_brick_rejections > 0:
                    warnings.warn(f'Failed to generate a valid brick after {generation_num + 1} attempts.\n'
                                  f'Last generated brick: {brick}\n'
                                  f'Reasons for rejection: {rejection_reasons}\n'
                                  f'Lego structure: {lego.to_txt()}\n')
                break

            # Reset if brick is invalid
            self.llm.rollback_to_saved_state()
            rejection_reasons.update([add_brick_result])
            rejected_bricks.add(brick)

            if add_brick_result == 'already_rejected':  # Increase temperature if brick has already been generated and rejected
                temperature = min(self.max_temperature, temperature + self.temperature_increase)

        return brick, rejection_reasons

    @staticmethod
    def _try_adding_brick(brick_str: str, lego: LegoStructure, rejected_bricks: set[str]) -> str:
        """
        Tries to add the brick, represented by a string, to the given LEGO structure.
        Returns the result: 'success' if the add was successful, and the failure reason otherwise.
        """
        if brick_str in rejected_bricks:
            return 'already_rejected'

        try:
            brick = LegoBrick.from_txt(brick_str)
        except ValueError:  # Brick is badly formatted
            return 'ill_formatted'
        try:
            _ = brick.brick_id
        except ValueError:  # Brick ID is not in library
            return 'not_in_library'

        if not lego.brick_in_bounds(brick):
            return 'out_of_bounds'
        if lego.brick_collides(brick):
            return 'collision'
        return 'success'

    def generate_brick(self, prompt: str | None = None, temperature: float | None = None) -> str:
        if temperature is None:
            temperature = self.temperature
        if self.use_logit_masking:
            return self._generate_brick_with_logit_masking(prompt, temperature)
        else:
            return self._generate_brick_no_logit_masking(prompt, temperature)

    def _generate_brick_no_logit_masking(
            self,
            prompt: str | None = None,
            temperature: float | None = None,
    ) -> str:
        """
        Generates a LEGO brick in txt format without logit masking.
        :param prompt: The prompt to be given to the LLM preceding brick generation.
        :return: A LEGO brick in txt format, or the empty string if generation is finished.
        """
        if temperature is None:
            temperature = self.temperature

        result_ids = self.llm(
            prompt,
            return_as_ids=True,
            max_new_tokens=10,
            temperature=temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return self.llm.tokenizer.decode(result_ids, skip_special_tokens=True)

    def _generate_brick_with_logit_masking(
            self,
            prompt: str | None = None,
            temperature: float | None = None,
    ) -> str:
        """
        Generates a LEGO brick in txt format, using logit masking to enforce compliance with the LEGO brick syntax.
        WARNING: Assumes each number in the brick dimensions and positions is represented by 1 token.
        :param prompt: The prompt to be given to the LLM preceding brick generation.
        :return: A LEGO brick in txt format, or the empty string if generation is finished.
        """
        if temperature is None:
            temperature = self.temperature

        allowed_dims = tuple(str(i) for i in range(1, max_brick_dimension + 1))
        allowed_posns = tuple(str(i) for i in range(self.world_dim))

        # Generate tokens one by one to fit the format "hxw (x,y,z)\n"
        result_ids = []
        for allowed_strs in [
            allowed_dims + (self.llm.tokenizer.eos_token,), ('x',), allowed_dims,
            (' (',), allowed_posns, (',',), allowed_posns, (',',), allowed_posns, (')\n',),
        ]:
            next_token_id = self.llm(
                prompt,
                return_as_ids=True,
                max_new_tokens=1,
                temperature=temperature,
                top_k=None,
                top_p=None,
                logits_processor=self._build_allow_tokens_logits_processor(allowed_strs)
            )[0]
            result_ids.append(next_token_id)

            if next_token_id == self.llm.tokenizer.eos_token_id:  # Generation is finished
                break
            if prompt is not None:
                prompt = None  # Only use prompt on first iteration; continue generation thereafter

        return self.llm.tokenizer.decode(result_ids, skip_special_tokens=True)

    @functools.cache
    def _build_allow_tokens_logits_processor(self, allowed_strs: tuple[str]) -> LogitsProcessorList:
        """
        Builds a logits processor that constrains the next token to be one of the allowed strings.
        """
        return LogitsProcessorList(
            [PrefixConstrainedLogitsProcessor(self._build_allowed_token_ids_fn(allowed_strs), num_beams=1)]
        )

    @functools.cache
    def _build_allowed_token_ids_fn(self, allowed_strs: tuple[str]) -> Callable[[int, torch.Tensor], list[int]]:
        """
        Builds a function that returns a set of allowed token IDs, to be used by PrefixConstrainedLogitsProcessor.
        """
        allowed_tokens = [self.llm.tokenizer.tokenize(s) for s in allowed_strs]
        if not all(len(tokens) == 1 for tokens in allowed_tokens):
            raise ValueError('Each allowed string must tokenize to exactly 1 token')
        allowed_ids = self.llm.tokenizer.convert_tokens_to_ids(tokens[0] for tokens in allowed_tokens)

        def allowed_token_ids_fn(_: int, __: torch.Tensor) -> list[int]:
            return allowed_ids

        return allowed_token_ids_fn


def create_instruction(caption: str) -> str:
    instruction = ('Create a LEGO model of the input. Format your response as a list of bricks: '
                   '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
                   'Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n'
                   'All bricks are 1 unit tall.\n\n'
                   '### Input:\n'
                   f'{caption}')
    return instruction


def create_instruction_zero_shot(caption: str) -> str:
    zero_shot_instructions = (
        'Each line of your output should be a LEGO brick in the format `<brick dimensions> <brick position>`. For example:\n'
        '2x4 (2,1,0)\n'
        'DO NOT output any other text. Only output LEGO bricks. The first brick should have a z-coordinate of 0.'
    )
    return '\n\n'.join([create_instruction(caption), zero_shot_instructions])


_few_shot_examples_filename = Path(__file__).parent / 'few_shot_examples.json'
with open(_few_shot_examples_filename) as f:
    _few_shot_examples = json.load(f)


def create_instruction_few_shot(caption: str) -> str:
    example_prompt = 'Here are some example LEGO models:'
    example_instructions = '\n\n'.join(_create_example_instruction(x) for x in _few_shot_examples)
    few_shot_instructions = (
        'Do NOT copy the examples, but create your own LEGO model for the following input.\n\n'
        '### Input:\n'
        f'{caption}\n\n'
        '### Output:\n'
    )
    return '\n\n'.join([create_instruction_zero_shot(caption), example_prompt,
                        example_instructions, few_shot_instructions])


def _create_example_instruction(x: dict) -> str:
    return f'### Input:\n{x["caption"]}\n\n### Output:\n{x["lego"]}'


def _remove_all_bricks_after_first_unstable_brick(lego: LegoStructure) -> LegoStructure:
    """
    Removes all bricks starting from the first unstable brick. Repeats this process until the lego is stable.
    """
    while True:
        if lego.is_stable():
            return lego
        scores = lego.stability_scores()
        first_unstable_brick_idx = next((i for i, brick in enumerate(lego.bricks)
                                         if np.any(scores[brick.slice] >= 1)), -1)
        lego = LegoStructure(lego.bricks[:first_unstable_brick_idx])
