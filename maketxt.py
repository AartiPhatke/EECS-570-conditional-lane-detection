import os

# where the JSONs live
LANE3D_ROOT = "images/lane3d_300"

# where the actual images live
IMAGES_ROOT = "images/images"

# where we will write txt files (already exists for you)
OUTPUT_DIR = "images/list"


def make_split_list(split_name, json_root, img_subdir, txt_name=None):
    """
    split_name: just for printing/logging
    json_root:  folder that contains the .json annotations
    img_subdir: relative path under IMAGES_ROOT where the .jpg live
                e.g. "training", "validation"
    txt_name:   output txt file name; if None, use f"{split_name}_list.txt"
    """
    if txt_name is None:
        txt_name = f"{split_name}_list.txt"

    json_root = os.path.join(LANE3D_ROOT, json_root)
    img_root = os.path.join(IMAGES_ROOT, img_subdir)
    out_path = os.path.join(OUTPUT_DIR, txt_name)

    print(f"Creating {out_path} from JSONs in {json_root}")

    count = 0
    missing = 0

    with open(out_path, "w") as f:
        for root, _, files in os.walk(json_root):
            for file in files:
                if not file.endswith(".json"):
                    continue

                # relative path from json_root, e.g. "segment-xxx"
                rel_dir = os.path.relpath(root, json_root)

                # same filename but .jpg instead of .json
                img_name = file.replace(".json", ".jpg")

                img_path = os.path.join(img_root, rel_dir, img_name)

                if os.path.exists(img_path):
                    f.write(img_path + "\n")
                    count += 1
                else:
                    print(f"WARNING: missing image for {os.path.join(root, file)}")
                    missing += 1

    print(f"{split_name}: wrote {count} lines, {missing} missing images\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) training: lane3d_300/training JSON -> images/images/training JPG
    make_split_list(
        split_name="training",
        json_root="training",
        img_subdir="training",
        txt_name="training_list.txt",
    )

    # 2) validation: lane3d_300/validation JSON -> images/images/validation JPG
    make_split_list(
        split_name="validation",
        json_root="validation",
        img_subdir="validation",
        txt_name="validation_list.txt",
    )

    # 3) intersection test set (if you want a separate txt for that)
    # JSONs: lane3d_300/test/intersection_case
    # Images: usually come from validation images (you can change if needed)
    make_split_list(
        split_name="intersection",
        json_root=os.path.join("test", "intersection_case"),
        img_subdir="validation",          # or "training" if that’s where they are
        txt_name="intersection_list.txt",
    )

    print("Done.")


if __name__ == "__main__":
    main()
