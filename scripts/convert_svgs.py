import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import re

def parse_dimensions(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        w_str = root.attrib.get('width', '')
        h_str = root.attrib.get('height', '')
        
        def extract_val(s):
            m = re.match(r'([0-9.]+)', s)
            return float(m.group(1)) if m else 0.0

        w = extract_val(w_str)
        h = extract_val(h_str)
        
        if w == 0 or h == 0:
            viewbox = root.attrib.get('viewBox', '')
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    w = float(parts[2])
                    h = float(parts[3])
        return w, h
    except Exception as e:
        print(f"Error parsing {svg_path}: {e}")
        return 0.0, 0.0

def convert_svgs(target_dir, out_format="pdf"):
    target_dir = Path(target_dir)
    
    if not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    svg_files = list(target_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {target_dir}")
        return

    print(f"Found {len(svg_files)} SVG files.")
    
    # Pass 1: Find max dimensions
    dimensions = {}
    max_width, max_height = 0.0, 0.0
    
    for svg_path in svg_files:
        w, h = parse_dimensions(svg_path)
        dimensions[svg_path] = (w, h)
        if w > max_width: max_width = w
        if h > max_height: max_height = h
        print(f"{svg_path.name}: {w} x {h}")
        
    print(f"Max dimensions: {max_width} x {max_height}")
    
    # Pass 2: Convert with padding
    for svg_path in svg_files:
        out_path = svg_path.with_suffix(f".{out_format}")
        w, h = dimensions[svg_path]
        
        top_pad = (max_height - h) / 2.0
        left_pad = (max_width - w) / 2.0
        
        # Build command for rsvg-convert
        cmd = [
            "rsvg-convert", "-f", out_format, 
            "-o", str(out_path), 
            "--page-width", f"{max_width}pt", 
            "--page-height", f"{max_height}pt",
            "--top", f"{top_pad}pt",
            "--left", f"{left_pad}pt",
            "-b", "white",
            str(svg_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Converted: {svg_path.name} -> {out_path.name} (centered)")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {svg_path.name}: {e}")
        except FileNotFoundError:
            print("Error: 'rsvg-convert' tool not found. Please install librsvg2-bin (or equivalent).")
            return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert SVGs to specified format with uniform canvas sizes, centered.")
    parser.add_argument("target_dir", type=str, nargs="?", 
                        default="outputs/figures/final_model_trees",
                        help="Directory containing SVG files (default: outputs/figures/final_model_trees)")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "ps", "eps", "svg"],
                        help="Output format (default: pdf)")
    args = parser.parse_args()
    
    target = Path(args.target_dir)
    if not target.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        target = base_dir / target
        
    convert_svgs(target, args.format)
