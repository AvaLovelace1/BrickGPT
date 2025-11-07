import sys
from pathlib import Path

# Add path to ImportLDraw module
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from brickgpt.render_bricks import main as render_bricks_main

if __name__ == '__main__':
    render_bricks_main()
