import os
import json
import cv2

# Base folders
JSON_BASE = "/scratch/engin_root/engin1/aphatke/conditional-lane-detection/images/lane3d_300/test/intersection_case"
IMG_BASE  = "/scratch/engin_root/engin1/aphatke/conditional-lane-detection/images/images/training"

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

        # Expect uv = [xs[], ys[]]
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

    # Write .lines.txt
    with open(txt_out, "w") as f:
        if results:
            f.write("\n".join(results))

    print(f"Saved: {txt_out}")
    return results


def visualize(img_path, lanes, out_img):
    """Draw the lanes on the image for sanity checking."""
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR loading:", img_path)
        return

    for lane in lanes:
        pts = list(map(float, lane.split(" ")))
        for i in range(0, len(pts), 2):
            x = int(pts[i])
            y = int(pts[i+1])
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

    cv2.imwrite(out_img, img)
    print("Preview saved:", out_img)


def find_matching_image(json_path):
    """
    Convert:
      images/lane3d_300/test/intersection_case/.../123456.json
    To:
      images/images/.../123456.jpg
    """
    filename = os.path.basename(json_path).replace(".json", ".jpg")

    # The segment folder is identical aside from the base root
    # We extract "segment-xxxx/...jpg"
    pieces = json_path.split("intersection_case/")[-1]
    folder = os.path.dirname(pieces)
    return os.path.join(IMG_BASE, folder, filename)


if __name__ == "__main__":
    print("\n=== Processing All Intersection JSON Files ===\n")

    for root, _, files in os.walk(JSON_BASE):
        for file in files:
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            print("\n--- JSON:", json_path)

            img_path = find_matching_image(json_path)
            if not os.path.exists(img_path):
                print("❌ Matching image not found:", img_path)
                continue
            print("✓ Found image:", img_path)

            txt_out = json_path.replace(".json", ".lines.txt")
            preview_out = json_path.replace(".json", "_preview.jpg")

            lanes = convert_single(json_path, txt_out)

            if lanes:
                visualize(img_path, lanes, preview_out)
            else:
                print("⚠️ No valid lanes after filtering.")

    print("\n=== DONE ===")
