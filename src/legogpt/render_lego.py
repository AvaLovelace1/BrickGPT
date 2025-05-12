import argparse
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import bpy

# Add path to ImportLDraw module
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

import ImportLDraw
from ImportLDraw.loadldraw.loadldraw import Options, Configure, loadFromFile, FileSystem


def render_lego(
        in_file: str,
        out_file: str,
        reposition_camera: bool = True,
        square_image: bool = True,
        instructions_look: bool = False,
        fov: float = 45,
        img_resolution: int = 512,
) -> None:
    in_file = os.path.abspath(in_file)
    out_file = os.path.abspath(out_file)

    # Set the path to the ImportLDraw and LDraw libraries
    plugin_path = Path(ImportLDraw.__file__).parent
    ldraw_lib_path = os.environ.get('LDRAW_LIBRARY_PATH')
    if not ldraw_lib_path or not os.path.exists(ldraw_lib_path):
        # Default path to LDraw library is home directory
        ldraw_lib_path = Path.home() / 'ldraw'
    ldraw_lib_path = os.path.abspath(ldraw_lib_path)

    # Initialize bpy
    with stdout_redirected(os.devnull):
        bpy.data.scenes[0].render.engine = 'CYCLES'

        # Set the device_type based on platform
        if sys.platform == 'darwin':  # macOS
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
        else:  # Linux/Windows
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Set the device and feature set
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 512

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)
        for d in bpy.context.preferences.addons['cycles'].preferences.devices:
            d['use'] = 0
            if sys.platform == 'darwin':  # macOS
                if d['name'].startswith('Apple'):  # Apple GPU
                    d['use'] = 1
            else:  # Linux/Windows
                if d['name'].startswith('NVIDIA'):  # NVIDIA GPU
                    d['use'] = 1
            print(d['name'], d['use'])

    # Remove all objects but keep the camera
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    Options.ldrawDirectory = ldraw_lib_path
    Options.instructionsLook = instructions_look
    Options.useLogoStuds = True
    Options.useUnofficialParts = True
    Options.gaps = True
    Options.studLogoDirectory = os.path.join(plugin_path, 'studs')
    Options.LSynthDirectory = os.path.join(plugin_path, 'lsynth')
    Options.verbose = 0
    Options.overwriteExistingMaterials = True
    Options.overwriteExistingMeshes = True
    Options.scale = 0.01
    Options.createInstances = True  # Multiple bricks share geometry (recommended)
    Options.removeDoubles = True  # Remove duplicate vertices (recommended)
    Options.positionObjectOnGroundAtOrigin = True  # Centre the object at the origin, sitting on the z=0 plane
    Options.flattenHierarchy = False  # All parts are under the root object - no sub-models
    Options.edgeSplit = True  # Add the edge split modifier
    Options.addBevelModifier = True  # Adds a bevel modifier to each part (for rounded edges)
    Options.bevelWidth = 0.5  # Bevel width
    Options.addEnvironmentTexture = True
    Options.scriptDirectory = os.path.join(plugin_path, 'loadldraw')
    Options.addWorldEnvironmentTexture = True  # Add an environment texture
    Options.addGroundPlane = True  # Add a ground plane
    Options.setRenderSettings = True  # Set render percentage, denoising
    Options.removeDefaultObjects = True  # Remove cube and lamp
    Options.positionCamera = reposition_camera  # Reposition the camera to a good place to see the scene
    Options.cameraBorderPercent = 0.05  # Add a percentage border around the repositioned camera

    Configure()
    loadFromFile(None, FileSystem.locate(in_file))

    if square_image:
        bpy.context.scene.render.resolution_x = img_resolution
        bpy.context.scene.render.resolution_y = img_resolution
        bpy.context.scene.camera.data.angle = math.radians(fov)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = out_file

    # Redirect stdout to suppress the verbose render output
    with stdout_redirected(os.devnull):
        bpy.ops.render.render(write_still=True)


@contextmanager
def stdout_redirected(to: str):
    """
    Redirects stdout to a file.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to_file):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to_file.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(old_stdout)  # restore stdout.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to LDR file')
    parser.add_argument('--out_file', type=str, help='Path to output image file')
    args = parser.parse_args()

    # Get the absolute path of the input file
    render_lego(args.in_file, args.out_file, square_image=True, instructions_look=False)
    print(f'Rendered image to {args.out_file}')


if __name__ == '__main__':
    main()
