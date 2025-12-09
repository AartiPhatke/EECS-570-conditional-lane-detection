import os
import json
import cv2


JSON_FILE = "images/images/training/segment-16102220208346880_1420_000_1440_000_with_camera_labels/155423288515811300.jpg"
IMG_FILE  = "/scratch/engin_root/engin1/aphatke/conditional-lane-detection/images/images/training/...YOUR_FILE.jpg"


OUTPUT_TXT = JSON_FILE.replace(".json", ".lines.txt")
OUTPUT_IMG = "single_label_preview.jpg"

IMG_W, IMG_H = 1920, 1280
MIN_VISIBLE_Y = 270  # remove sky points


def convert_single(json_path, txt_out):
    """Convert one OpenLane JSON → .lines.txt"""
    with open(json_path, "r") as f:
        data = json.load(f)

    lanes = data.get("lane_lines", [])
    results = []

    for lane in lanes:
        uv = lane.get("uv", [])
        vis = lane.get("visibility", [])

        if len(uv) != 2:
            continue

        xs = uv[0]
        ys = uv[1]

        if len(vis) != len(xs):
            vis = [1.0] * len(xs)

        coords = []

        for x, y, v in zip(xs, ys, vis):
            x = float(x)
            y = float(y)

            if v <= 0: continue
            if y < MIN_VISIBLE_Y: continue
            if not (0 <= x <= IMG_W): continue
            if not (0 <= y <= IMG_H): continue

            coords.append(f"{x:.4f} {y:.4f}")

        if len(coords) >= 2:
            results.append(" ".join(coords))

    with open(txt_out, "w") as f:
        if results:
            f.write("\n".join(results))

    print(f"Saved: {txt_out}")
    return results


def visualize(img_path, lanes, out_img):
    """Draw the lanes on the image for sanity checking."""
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot load image:", img_path)
        return

    for lane in lanes:
        pts = lane.split(" ")
        pts = list(map(float, pts))

        for i in range(0, len(pts), 2):
            x = int(pts[i])
            y = int(pts[i+1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv2.imwrite(out_img, img)
    print(f"Saved visualization: {out_img}")


if __name__ == "__main__":
    print("Converting single JSON...")
    lanes = convert_single(JSON_FILE, OUTPUT_TXT)

    if not lanes:
        print("No lanes produced! Check filters.")
    else:
        print("Visualizing...")
        visualize(IMG_FILE, lanes, OUTPUT_IMG)

    print("\nDONE.")
