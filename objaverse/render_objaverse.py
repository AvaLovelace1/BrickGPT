
import os
import subprocess
from pathlib import Path
import argparse

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = SCRIPT_DIR / "assets"

"""
Scans assets directory for .ldr files and renders them to .png if the png doesn't exist
"""
def render_assets(force_all=False):
    if not ASSETS_DIR.exists():
        print(f"Error: assets directory not found at {ASSETS_DIR}")
        return

    # Find all .ldr files
    ldr_files = sorted(list(ASSETS_DIR.rglob("*.ldr")))
    print(f"Found {len(ldr_files)} LDR files to check for rendering.")

    # Prepare environment with local ldraw library
    env = os.environ.copy()
    ldraw_path = SCRIPT_DIR.parent / "ldraw"
    if ldraw_path.exists():
        env["LDRAW_LIBRARY_PATH"] = str(ldraw_path)
    
    count = 0
    
    for ldr_path in ldr_files:
        png_path = ldr_path.with_suffix(".png")
        
        if png_path.exists() and not force_all:
            print(f"[SKIP] {ldr_path.parent.name}/{png_path.name}")
            continue
            
        print(f"[RENDER] {ldr_path.parent.name}/{ldr_path.name}...")
        
        # Construct command: uv run render_bricks --in_file <ldr> --out_file <png>
        cmd = [
            "uv", "run", "render_bricks",
            "--in_file", str(ldr_path),
            "--out_file", str(png_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=SCRIPT_DIR.parent, env=env, stdout=subprocess.DEVNULL)
            print(f"  -> Success")
            count += 1
        except subprocess.CalledProcessError as e:
            print(f"  -> FAILED: {e}")
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

    print(f"\nRendered {count} new images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Objaverse LDR files to PNG.")
    parser.add_argument("--force", action="store_true", help="Force re-rendering of existing PNGs")
    args = parser.parse_args()
    
    render_assets(force_all=args.force)
