import os
import json


JSON_ROOT = "images/lane3d_300" 
IMAGE_ROOT = "images/images"     

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
            
 
            lanes = data.get('lane_lines', [])
            
  
            rel_dir = os.path.relpath(root, json_path)
            txt_filename = file.replace(".json", ".lines.txt")
            output_path = os.path.join(img_path_root, rel_dir, txt_filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            valid_lines_for_file = []

            for lane in lanes:
  
                points = lane.get('uv', [])
                visibility = lane.get('visibility', [])
                
  
                if not points:
                    continue

                line_coords = []
                
                for i, p in enumerate(points):
        
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue 
 
                    if i < len(visibility) and visibility[i] <= 0:
                        continue

                    try:
                        x = float(p[0])
                        y = float(p[1])

                        if x < 0 or y < 0:
                            continue

                        line_coords.append(f"{x:.4f} {y:.4f}")
                    except (ValueError, IndexError):
                        continue
                

                if len(line_coords) >= 2:
                    valid_lines_for_file.append(" ".join(line_coords))
            
            with open(output_path, 'w') as f_out:
                if valid_lines_for_file:
                    f_out.write("\n".join(valid_lines_for_file))
            
            processed_count += 1

    print(f"Done {split_name}: Processed {processed_count} files.")

if __name__ == "__main__":
    convert_segment("Validation", "validation", "validation")