import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import transformers
from PIL import Image
from brickgpt.models import BrickGPT, BrickGPTConfig


class Demo:
    def __init__(self, model_cfg: BrickGPTConfig, flagging_dir: str):
        self.flagging_dir = '/data/apun/brickgpt_demo_out'
        os.makedirs(self.flagging_dir, exist_ok=True)
        self.generator = BrickGenerator(flagging_dir, model_cfg)

        # Inputs
        self.in_prompt = gr.Textbox(label='Input prompt', info='Text prompt for which to generate a brick structure.',
                                    max_length=2048)
        self.in_optout = gr.Checkbox(label='Do not save my data',
                                     info='We may collect inputs and outputs to help us improve the model. '
                                          'Your data will never be shared or used for any other purpose. '
                                          'If you wish to opt out of data collection, check this box.')
        self.in_temperature = gr.Slider(0.01, 2.0, value=model_cfg.temperature, step=0.01, precision=2,
                                        label='Temperature', info=get_help_string('temperature'))
        self.in_seed = gr.Number(value=42, label='Seed', info='Random seed for generation.',
                                 precision=0, minimum=0, maximum=2 ** 32 - 1, step=1)
        self.in_bricks = gr.Number(value=model_cfg.max_bricks, label='Max bricks', info=get_help_string('max_bricks'),
                                   precision=0, minimum=1, step=1)
        self.in_rejections = gr.Number(value=model_cfg.max_brick_rejections, label='Max brick rejections',
                                       info=get_help_string('max_brick_rejections'), precision=0, minimum=0, step=1)
        self.in_regenerations = gr.Number(value=model_cfg.max_regenerations, label='Max regenerations',
                                          info=get_help_string('max_regenerations'), precision=0, minimum=0, step=1)
        self.generate_btn = gr.Button('Generate!', variant='primary')

        # Outputs
        self.out_img = gr.Image(label='Output image', format='png')
        self.out_txt = gr.Textbox(label='Output bricks', lines=5, max_lines=5, show_copy_button=True,
                                  info='The brick structure in text format. Each line of the form "hxw (x,y,z)" represents a '
                                       '1-unit-tall rectangular brick with dimensions hxw placed at coordinates (x,y,z).')
        self.out_flag_data = gr.JSON(visible=False)
        self.flag_bad_btn = gr.Button('Rate as Bad 😞', size='md')
        self.flag_okay_btn = gr.Button('Rate as Okay 😐', size='md')
        self.flag_great_btn = gr.Button('Rate as Great 😄', size='md')
        self.report_issue_btn = gr.Button('Report an issue', size='sm',
                                          link='https://github.com/AvaLovelace1/BrickGPT/issues/new/choose')

        self.demo = gr.Blocks(theme=gr.themes.Monochrome())

    def _render(self) -> None:
        with self.demo:
            gr.Markdown('# BrickGPT Demo')
            gr.Markdown(
                'This is the official demo for **[BrickGPT](https://avalovelace1.github.io/BrickGPT/)**, the first approach for generating physically stable toy brick structures from text prompts.\n\n'
                'BrickGPT is restricted to creating structures made of 1-unit-tall cuboid bricks on a 20x20x20 grid. It was trained on a dataset of 21 object categories: '
                '*basket, bed, bench, birdhouse, bookshelf, bottle, bowl, bus, camera, car, chair, guitar, jar, mug, piano, pot, sofa, table, tower, train, vessel.* '
                'Performance on prompts from outside these categories may be limited. This demo does not include texturing or coloring.')

            with gr.Row():
                with gr.Column():
                    self.in_prompt.render()
                    self.in_optout.render()
                    with gr.Accordion(label="Advanced options", open=False):
                        self.in_temperature.render()
                        self.in_seed.render()
                        self.in_bricks.render()
                        self.in_rejections.render()
                        self.in_regenerations.render()
                    self.generate_btn.render()

                with gr.Column():
                    self.out_img.render()
                    self.out_txt.render()
                    self.out_flag_data.render()
                    with gr.Row():
                        self.flag_bad_btn.render()
                        self.flag_okay_btn.render()
                        self.flag_great_btn.render()
                    self.report_issue_btn.render()

            examples = get_examples()
            dummy_name = gr.Textbox(visible=False, label='Name')
            dummy_out_img = gr.Image(visible=False, label='Result')
            gr.Examples(
                examples=[[name, example['prompt'], example['temperature'], example['seed'], example['output_img']]
                          for name, example in examples.items()],
                inputs=[dummy_name, self.in_prompt, self.in_temperature, self.in_seed, dummy_out_img],
                outputs=[self.out_img, self.out_txt, self.out_flag_data],
                fn=lambda *args: (args[-1], examples[args[0]]['output_txt'], examples[args[0]]),
                run_on_click=True,
            )

    def _bind_events(self) -> None:
        with self.demo:
            self.generate_btn.click(
                fn=self.generator.generate_bricks,
                inputs=[
                    self.in_prompt,
                    self.in_optout,
                    self.in_temperature,
                    self.in_seed,
                    self.in_bricks,
                    self.in_rejections,
                    self.in_regenerations,
                ],
                outputs=[self.out_img, self.out_txt, self.out_flag_data],
            )

            for btn, rating in [
                (self.flag_bad_btn, 'bad'),
                (self.flag_okay_btn, 'okay'),
                (self.flag_great_btn, 'great'),
            ]:
                btn.click(
                    fn=self._flag,
                    inputs=[self.out_flag_data, gr.State(rating), self.in_optout],
                )

    def _flag(
            self,
            flag_data: dict[str, Any] | None,
            rating: str,
            do_not_save_data: bool,
    ) -> None:
        if do_not_save_data:
            raise gr.Error('Uncheck "Do not save my data" to enable feedback collection.')
        if flag_data is None:
            raise gr.Error('Can\'t rate an empty output.')
        if 'uid' not in flag_data:
            assert flag_data.get('example', False)
            flag_data['uid'] = f'example_{uuid.uuid4()}'

        flag_data['rating'] = rating

        print(f'Logging flagged data: {flag_data}')
        out_filename = os.path.join(self.flagging_dir, f'{flag_data["uid"]}.json')
        with open(out_filename, 'w') as f:
            json.dump(flag_data, f)
        print(f'Saved flagged data to {out_filename}.')

        gr.Info('Your feedback has been recorded. Thank you!')

    def launch(self) -> None:
        self._bind_events()
        self._render()
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
    def __init__(self, flagging_dir: str, model_cfg: BrickGPTConfig):
        self.flagging_dir = flagging_dir
        os.makedirs(self.flagging_dir, exist_ok=True)
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
    ) -> tuple[Image.Image, str, dict[str, Any]]:
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
        generation_time = time.time() - start_time
        print(f'Finished generation in {generation_time:.1f}s!')

        output_uuid = str(uuid.uuid4())
        output_txt = output['bricks'].to_txt()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write output LDR to tmp file
            ldr_filename = os.path.join(tmp_dir, f'{output_uuid}.ldr')
            with open(ldr_filename, 'w') as f:
                f.write(output['bricks'].to_ldr())

            # Render brick model to tmp image
            print('Rendering image...')
            img_filename = os.path.join(tmp_dir, f'{output_uuid}.png')
            subprocess.run(['python', self.render_bricks_script, '--in_file', ldr_filename, '--out_file', img_filename],
                           check=True)  # Run render as a subprocess to prevent issues with Blender
            rendering_time = time.time() - start_time - generation_time
            print(f'Finished rendering in {rendering_time:.1f}s!')

            # Load image
            img = Image.open(img_filename)

        flag_data = {
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
            out_filename = os.path.join(self.flagging_dir, f'{flag_data["uid"]}.json')
            with open(out_filename, 'w') as f:
                json.dump(flag_data, f)
            print(f'Saved data to {out_filename}.')

        return img, output_txt, flag_data


my_demo = Demo(BrickGPTConfig(max_regenerations=5, device='cuda'),
               flagging_dir='/data/apun/brickgpt_demo_out')
demo = my_demo.demo  # __main__ needs a "demo" attribute for Gradio hot reloading to work
my_demo.launch()
