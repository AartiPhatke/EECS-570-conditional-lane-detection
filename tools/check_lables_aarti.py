import cv2
import os
import sys

# CHANGE THESE TWO PATHS TO A REAL FILE YOU HAVE
img_path = "images/images/training/segment-8487809726845917818_4779_870_4799_870_with_camera_labels/150716681676958000.jpg"
txt_path = "images/images/training/segment-8487809726845917818_4779_870_4799_870_with_camera_labels/150716681676958000.lines.txt"

# If paths are relative, add your root
root = "/scratch/engin_root/engin1/aphatke/conditional-lane-detection/"
full_img_path = os.path.join(root, img_path)
full_txt_path = os.path.join(root, txt_path)

print(f"Checking: {full_img_path}")

img = cv2.imread(full_img_path)
if img is None:
    print("Error: Could not read image.")
    sys.exit()

with open(full_txt_path, 'r') as f:
    lines = f.readlines()

print(f"Found {len(lines)} lanes in text file.")

for line in lines:
    coords = list(map(float, line.strip().split()))
    
    # If len is 4 (x1 y1 x2 y2), it's a straight line (BAD for curves)
    if len(coords) == 4:
        print("WARNING: Lane has only 2 points! (Straight line)")
    else:
        print(f"Lane has {len(coords)//2} points.")

    # Draw the points
    for i in range(0, len(coords), 2):
        x, y = int(coords[i]), int(coords[i+1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # Red Dots

# Save result
cv2.imwrite("sanity_check.jpg", img)
print("Saved sanity_check.jpg. Download and inspect it.")