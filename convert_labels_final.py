import os
import json

# ================= CONFIGURATION =================
# Update these to match your actual folder structure
JSON_ROOT = "images/lane3d_300" 
IMAGE_ROOT = "images/images"     
# =================================================

def convert_segment(split_name, json_subdir, img_subdir):
    json_path = os.path.join(JSON_ROOT, json_subdir)
    img_path_root = os.path.join(IMAGE_ROOT, img_subdir)
    
    print(f"--- Processing {split_name} ---")
    
    processed_count = 0
    
    for root, _, files in os.walk(json_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            
            full_json_path = os.path.join(root, file)
            
            try:
                with open(full_json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skipping corrupt JSON: {file}")
                continue
            
            # Extract Lane Lines
            lanes = data.get('lane_lines', [])
            
            # Prepare output .lines.txt path
            rel_dir = os.path.relpath(root, json_path)
            txt_filename = file.replace(".json", ".lines.txt")
            output_path = os.path.join(img_path_root, rel_dir, txt_filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            valid_lines_for_file = []

            for lane in lanes:
                # 1. USE 'uv' (Pixel coordinates), NOT 'xyz'
                # CondLaneNet works on images, so we need pixels.
                points = lane.get('uv', [])
                visibility = lane.get('visibility', [])
                
                # Check if we have points
                if not points:
                    continue

                line_coords = []
                
                # Iterate through points
                for i, p in enumerate(points):
                    # --- SAFETY CHECK 1: Malformed Data ---
                    # Ensure p is a list and has at least 2 items (x, y)
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue 
                    
                    # --- SAFETY CHECK 2: Visibility ---
                    # If visibility data exists, skip invisible points (0.0)
                    if i < len(visibility) and visibility[i] <= 0:
                        continue

                    try:
                        x = float(p[0])
                        y = float(p[1])
                        
                        # --- SAFETY CHECK 3: Image Bounds (Optional but good) ---
                        # Ignore points that are obviously garbage (negative pixels)
                        if x < 0 or y < 0:
                            continue

                        line_coords.append(f"{x:.4f} {y:.4f}")
                    except (ValueError, IndexError):
                        continue
                
                # Only write the lane if it has at least 2 valid points
                if len(line_coords) >= 2:
                    valid_lines_for_file.append(" ".join(line_coords))
            
            # Write to .lines.txt file
            # Even if empty, we create the file so the loader doesn't crash
            with open(output_path, 'w') as f_out:
                if valid_lines_for_file:
                    f_out.write("\n".join(valid_lines_for_file))
            
            processed_count += 1

    print(f"Done {split_name}: Processed {processed_count} files.")

if __name__ == "__main__":
    # convert_segment("Training", "training", "training")
    convert_segment("Validation", "validation", "validation")