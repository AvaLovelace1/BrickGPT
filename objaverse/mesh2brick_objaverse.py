
import argparse
import os
from pathlib import Path
from mesh2brick.mesh2brick import Mesh2Brick

SCRIPT_DIR = Path(__file__).parent
OBJAVERSE_DIR = SCRIPT_DIR / "assets"

def convert_objaverse_assets(resolution: int):
    if not OBJAVERSE_DIR.exists():
        print(f"Directory not found: {OBJAVERSE_DIR}")
        return

    glb_files = list(OBJAVERSE_DIR.glob("*.glb"))
    print(f"Found {len(glb_files)} files in {OBJAVERSE_DIR}")

    print(f"\n--- Processing resolution: {resolution} ---")
    
    # Create subfolder for this resolution
    res_dir = OBJAVERSE_DIR / f"res_{resolution}"
    res_dir.mkdir(exist_ok=True)
    print(f"Output directory: {res_dir}")
    
    converter = Mesh2Brick(world_dim=(resolution, resolution, resolution))

    for file_path in glb_files:
        filename = file_path.name
        print(f"Converting {filename}...")
    
        try:
            bricks = converter(str(file_path))
            
            # 1. Save TXT
            txt_output_path = res_dir / file_path.with_suffix(".txt").name
            txt_content = bricks.to_txt()
            with open(txt_output_path, "w") as f:
                f.write(txt_content)
            
            # 2. Save LDR
            ldr_output_path = res_dir / file_path.with_suffix(".ldr").name
            ldr_content = bricks.to_ldr()
            with open(ldr_output_path, "w") as f:
                f.write(ldr_content)
                
                
            print(f"Saved to {res_dir.name}/{ldr_output_path.name}")
        except Exception as e:
            print(f"FAILED to convert {filename} at resolution {resolution}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Objaverse assets to LEGO bricks.")
    parser.add_argument("--resolution", type=int, default=20, help="Voxel resolution (world_dim)")
    args = parser.parse_args()
    
    convert_objaverse_assets(args.resolution)
