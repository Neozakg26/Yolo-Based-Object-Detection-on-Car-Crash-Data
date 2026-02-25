#In this File I want to get a bddk tracking scene folder.
#  Divide it into mod 50
# rename the scenes accordingly per 50
import argparse
import shutil
from pathlib import Path


def total_files_in_directory(path_str):
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix == ".jpg")


def rename_files(path_str, output_str):
    path = Path(path_str)
    output = Path(output_str)

    folder = path.name
    prefix = folder.rsplit("-",1)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    output.mkdir(parents=True, exist_ok=True)

    # Collect all .jpg files and extract their id from "hash-hash-id.jpg"
    jpg_files = []
    for f in path.iterdir():
        if f.is_file() and f.suffix == ".jpg":
            stem = f.stem  # e.g. "0062298d-cbbec2cd-0000001"
            parts = stem.rsplit("-", 1)
            if len(parts) == 2:
                try:
                    file_id = int(parts[1])
                    jpg_files.append((file_id, f))
                except ValueError:
                    print(f"Skipping {f.name}: could not parse id from '{parts[1]}'")
            else:
                print(f"Skipping {f.name}: does not match 'hash-hash-id.jpg' format")

    # Sort by id so renaming is deterministic
    jpg_files.sort(key=lambda x: x[0])
    tuple_list = []

    for file_id, filepath in jpg_files:
        scene = file_id // 50
        local_id = file_id % 50
        new_name = f"{prefix[1]}_{scene}_{local_id:06d}.jpg"
        new_path = output / new_name
        results_set = (filepath,new_path)
        tuple_list.append(results_set)
        if len(tuple_list)==50:
            for src, dest in tuple_list:
                print(f"Now copying : {src.name} -> {dest}")
                shutil.copy2(src, dest)
            tuple_list.clear

    print(f"Copied {len(jpg_files)} files across {(jpg_files[-1][0] // 50) + 1 if jpg_files else 0} scenes to {output}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename BDDK tracking JPGs into scene-based naming.")
    parser.add_argument("--path", type=str, required=True, help="Folder containing hash-hash-id.jpg files")
    parser.add_argument("--output", type=str, required=True, help="Target output folder for renamed files")
    args = parser.parse_args()

    total = total_files_in_directory(args.path)
    print(f"Found {total} JPG files in {args.path}")

    rename_files(args.path, args.output)
