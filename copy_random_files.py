import os
import random
import shutil
import argparse

def copy_random_files(source_dir, dest_dir, num_files):
    """
    Copies a specified number of random files from a source directory to a destination directory.

    Args:
        source_dir (str): The path to the directory to copy files from.
        dest_dir (str): The path to the directory to copy files to.
        num_files (int): The number of random files to copy.
    """
    # --- 1. Validate the source directory ---
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # --- 2. Create the destination directory if it doesn't exist ---
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create destination directory '{dest_dir}'. {e}")
        return

    # --- 3. Get a list of all files in the source directory ---
    # os.path.isfile() ensures we don't accidentally include subdirectories
    try:
        all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    except OSError as e:
        print(f"Error: Could not read files from source directory '{source_dir}'. {e}")
        return
        
    # --- 4. Check if there are enough files to copy ---
    if not all_files:
        print(f"Warning: No files found in the source directory '{source_dir}'.")
        return

    if len(all_files) < num_files:
        print(f"Warning: Source directory has only {len(all_files)} files.")
        print(f"Copying all {len(all_files)} files instead of the requested {num_files}.")
        num_files_to_copy = len(all_files)
    else:
        num_files_to_copy = num_files

    # --- 5. Randomly select the files ---
    # random.sample ensures we get unique files without replacement
    try:
        files_to_copy = random.sample(all_files, num_files_to_copy)
    except ValueError:
        print("Error: Could not perform random sampling. Check file list.")
        return

    # --- 6. Copy the selected files ---
    print(f"Copying {len(files_to_copy)} random files from '{source_dir}' to '{dest_dir}'...")
    copied_count = 0
    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(dest_dir, filename)
        try:
            # shutil.copy2 preserves file metadata (like creation/modification time)
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        except Exception as e:
            print(f"\nError copying file '{filename}': {e}")

    print(f"\nSuccessfully copied {copied_count} out of {len(files_to_copy)} selected files.")

if __name__ == "__main__":
    # --- Set up the command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Copy a specific number of random files from a source to a destination folder.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("source", type=str, help="The source directory path.")
    parser.add_argument("destination", type=str, help="The destination directory path.")
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=50,
        help="The number of random files to copy (default: 50)."
    )

    args = parser.parse_args()

    # --- Call the main function with the parsed arguments ---
    copy_random_files(args.source, args.destination, args.num)