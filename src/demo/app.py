import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Sequence, Any

import gradio as gr
import torch
import transformers
from brickgpt.models import BrickGPT, BrickGPTConfig
from gradio import Component


class Demo:
    def __init__(self, output_dir: str, model_cfg: BrickGPTConfig):
        flagging_dir = '/data/apun/brickgpt_demo_out'
        self.generator = BrickGenerator(output_dir, flagging_dir, model_cfg)
        self.logger = Logger()

        # Define inputs and outputs
        in_prompt = gr.Textbox(label='Input prompt', info='Text prompt for which to generate a brick structure.',
                               max_length=2048)
        in_optout = gr.Checkbox(label='Do not save my data',
                                info='We may collect inputs and outputs to help us improve the model. '
                                     'Your data will never be shared or used for any other purpose. '
                                     'If you wish to opt out of data collection, check this box.')
        in_temperature = gr.Slider(0.01, 2.0, value=model_cfg.temperature, step=0.01, precision=2,
                                   label='Temperature', info=get_help_string('temperature'))
        in_seed = gr.Number(value=42, label='Seed', info='Random seed for generation.',
                            precision=0, minimum=0, maximum=2 ** 32 - 1, step=1)
        in_bricks = gr.Number(value=model_cfg.max_bricks, label='Max bricks', info=get_help_string('max_bricks'),
                              precision=0, minimum=1, step=1)
        in_rejections = gr.Number(value=model_cfg.max_brick_rejections, label='Max brick rejections',
                                  info=get_help_string('max_brick_rejections'), precision=0, minimum=0, step=1)
        in_regenerations = gr.Number(value=model_cfg.max_regenerations, label='Max regenerations',
                                     info=get_help_string('max_regenerations'), precision=0, minimum=0, step=1)
        out_img = gr.Image(label='Output image', format='png')
        out_txt = gr.Textbox(label='Output bricks', lines=5, max_lines=5, show_copy_button=True,
                             info='The brick structure in text format. Each line of the form "hxw (x,y,z)" represents a '
                                  '1-unit-tall rectangular brick with dimensions hxw placed at coordinates (x,y,z).')
        out_metadata = gr.JSON(label='out_metadata', visible=False)

        self.demo = gr.Interface(
            fn=self.generator.generate_bricks,
            title='BrickGPT Demo',
            description='This is the official demo for [BrickGPT](https://avalovelace1.github.io/BrickGPT/), the first approach for generating physically stable toy brick structures from text prompts.\n\n'
                        'BrickGPT is restricted to creating structures made of 1-unit-tall cuboid bricks on a 20x20x20 grid. It was trained on a dataset of 21 object categories: '
                        '*basket, bed, bench, birdhouse, bookshelf, bottle, bowl, bus, camera, car, chair, guitar, jar, mug, piano, pot, sofa, table, tower, train, vessel.* '
                        'Performance on prompts from outside these categories may be limited. This demo does not include texturing or coloring.\n\n'
                        'If you encounter any problems, [create an issue on GitHub](https://github.com/AvaLovelace1/BrickGPT/issues/new/choose).',
            inputs=[in_prompt, in_optout],
            additional_inputs=[in_temperature, in_seed, in_bricks, in_rejections, in_regenerations],
            outputs=[out_img, out_txt, out_metadata],
            flagging_options=[('Rate as Bad 😞', 'bad'),
                              ('Rate as Okay 😐', 'okay'),
                              ('Rate as Great 😄', 'great')],
            flagging_dir=flagging_dir,
            flagging_callback=self.logger,
            theme=gr.themes.Monochrome(),
        )

        with self.demo:
            examples = get_examples()
            dummy_name = gr.Textbox(visible=False, label='Name')
            dummy_out_img = gr.Image(visible=False, label='Result')
            gr.Examples(
                examples=[[name, example['prompt'], example['temperature'], example['seed'], example['output_img']]
                          for name, example in examples.items()],
                inputs=[dummy_name, in_prompt, in_temperature, in_seed, dummy_out_img],
                outputs=[out_img, out_txt, out_metadata],
                fn=lambda *args: (args[-1], examples[args[0]]['output_txt'], examples[args[0]]),
                run_on_click=True,
            )

    def launch(self) -> None:
        self.demo.queue().launch()


def get_help_string(field_name: str) -> str:
    """
    :param field_name: Name of a field in BrickGPTConfig.
    :return: Help string for the field.
    """
    data_fields = fields(BrickGPTConfig)
    name_field = next(f for f in data_fields if f.name == field_name)
    return name_field.metadata['help']


def get_examples(example_dir: str = str(Path(__file__).parent / 'examples')) -> dict[str, dict[str, str]]:
    examples_file = os.path.join(example_dir, 'examples.json')
    with open(examples_file) as f:
        examples = json.load(f)

    for example in examples.values():
        example['output_img'] = os.path.join(example_dir, example['output_img'])
    return examples


class BrickGenerator:
    def __init__(self, output_dir: str, flagging_dir: str, model_cfg: BrickGPTConfig):
        self.output_dir = output_dir
        self.flagging_dir = flagging_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model_cfg = model_cfg
        self.model = None

        self.render_bricks_script = str(Path(__file__).parent / 'render_bricks.py')
        self.save_data_dir = '/data/apun/brickgpt_demo_out'
        os.makedirs(self.save_data_dir, exist_ok=True)

    def generate_bricks(self, *args, **kwargs):
        try:
            return self._generate_bricks(*args, **kwargs)
        except torch.OutOfMemoryError:
            raise gr.Error('The model ran out of GPU memory. '
                           'Try reducing the "Max bricks" or "Max regenerations" parameters, or choose a different seed.')

    def _generate_bricks(
            self,
            prompt: str,
            do_not_save_data: bool,
            temperature: float | None,
            seed: int | None,
            max_bricks: int | None,
            max_brick_rejections: int | None,
            max_regenerations: int | None,
    ) -> tuple[str, str, dict[str, Any]]:
        self.model = BrickGPT(self.model_cfg)

        # Set model parameters
        if temperature is not None: self.model.temperature = temperature
        if max_bricks is not None: self.model.max_bricks = max_bricks
        if max_brick_rejections is not None: self.model.max_brick_rejections = max_brick_rejections
        if max_regenerations is not None: self.model.max_regenerations = max_regenerations
        if seed is not None: transformers.set_seed(seed)

        # Generate bricks
        print(f'Generating bricks for prompt: "{prompt}"')
        start_time = time.time()
        output = self.model(prompt)

        # Write output LDR to file
        output_uuid = str(uuid.uuid4())
        ldr_filename = os.path.join(self.output_dir, f'{output_uuid}.ldr')
        with open(ldr_filename, 'w') as f:
            f.write(output['bricks'].to_ldr())
        generation_time = time.time() - start_time
        output_txt = output['bricks'].to_txt()
        print(f'Finished generation in {generation_time:.1f}s!')

        # Render brick model to image
        print('Rendering image...')
        img_filename = os.path.join(self.output_dir, f'{output_uuid}.png')
        subprocess.run(['python', self.render_bricks_script, '--in_file', ldr_filename, '--out_file', img_filename],
                       check=True)  # Run render as a subprocess to prevent issues with Blender
        rendering_time = time.time() - start_time - generation_time
        print(f'Finished rendering in {rendering_time:.1f}s!')

        metadata = {
            'uid': output_uuid,
            'prompt': prompt,
            'temperature': self.model.temperature,
            'seed': seed,
            'max_bricks': self.model.max_bricks,
            'max_brick_rejections': self.model.max_brick_rejections,
            'max_regenerations': self.model.max_regenerations,
            'start_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'generation_time': generation_time,
            'rendering_time': rendering_time,
            'output_txt': output_txt,
        }

        if not do_not_save_data:
            out_filename = os.path.join(self.flagging_dir, f'{metadata["uid"]}.json')
            with open(out_filename, 'w') as f:
                json.dump(metadata, f)
            print(f'Saved data to {out_filename}.')

        return img_filename, output_txt, metadata


class Logger(gr.FlaggingCallback):
    def __init__(self):
        self.components = None
        self.flagging_dir = None

    def setup(
            self,
            components: Sequence[Component],
            flagging_dir: str | Path,
    ):
        self.components = components
        self.flagging_dir = str(flagging_dir)
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
            self,
            flag_data: list[Any],
            flag_option: str | None = None,
            username: str | None = None,
    ) -> int:
        components_and_data = zip(self.components, flag_data, strict=False)
        metadata = next(data for component, data in components_and_data
                        if component.label == 'out_metadata')

        if metadata is None:
            gr.Info('Can\'t flag an empty output.')
            return 0
        if metadata.get('example', False):
            return 0  # Do not log example data

        metadata['rating'] = flag_option

        print(f'Logging flagged data: {metadata}')
        out_filename = os.path.join(self.flagging_dir, f'{metadata["uid"]}.json')
        with open(out_filename, 'w') as f:
            json.dump(metadata, f)
        print(f'Saved flagged data to {out_filename}.')
        return 1


with tempfile.TemporaryDirectory() as tmp_dir:
    my_demo = Demo(tmp_dir, BrickGPTConfig(max_regenerations=5, device='cuda'))
    demo = my_demo.demo  # __main__ needs a "demo" attribute for Gradio hot reloading to work
    my_demo.launch()
