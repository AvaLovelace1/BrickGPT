
import os
from pathlib import Path
from mesh2brick.mesh2brick import Mesh2Brick

SCRIPT_DIR = Path(__file__).parent
OBJAVERSE_DIR = SCRIPT_DIR / "objaverse"

def convert_objaverse_assets():
    if not OBJAVERSE_DIR.exists():
        print(f"Directory not found: {OBJAVERSE_DIR}")
        return

    glb_files = list(OBJAVERSE_DIR.glob("*.glb"))
    print(f"Found {len(glb_files)} files in {OBJAVERSE_DIR}")

    converter = Mesh2Brick()

    for file_path in glb_files:
        filename = file_path.name
        print(f"Converting {filename}...")
    
        bricks = converter(str(file_path))
        
        # 1. Save TXT
        txt_output_path = file_path.with_suffix(".txt")
        txt_content = bricks.to_txt()
        with open(txt_output_path, "w") as f:
            f.write(txt_content)
        
        # 2. Save LDR
        ldr_output_path = file_path.with_suffix(".ldr")
        ldr_content = bricks.to_ldr()
        with open(ldr_output_path, "w") as f:
            f.write(ldr_content)
            
        print(f"Saved {txt_output_path.name} and {ldr_output_path.name}")

if __name__ == "__main__":
    convert_objaverse_assets()
