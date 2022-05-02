"""
# In this example, we use the voxel grid representation.
#
# This code unzip the downloaded dataset and format with following structure in a sequence directory:
#
# seq_name (e.g. zurich_city/11/a)
# ├── disparity
# │   ├── event
# │   │   ├── 000000.png
# │   │   └── ...
# │   └── timestamps.txt
# └── events
#     ├── left
#     │   ├── events.h5
#     │   └── rectify_map.h5
#     └── right
#         ├── events.h5
#         └── rectify_map.h5

"""
import os
import glob


def unzip_files(path, string="zurich_city"):
    """
    unzip files and save to new directory with format
    """
    no_replace = string.replace("_", "/")
    unzipped_dir = path.replace(".zip", "")
    new_dir = unzipped_dir.replace("_", "/")
    new_dir = new_dir.replace(no_replace, string)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        os.system("unzip " + path + " -d " + new_dir)
    print("File unzipped and saved to: " + new_dir)


def copy_files(path, string="zurich_city"):
    filename = path.split("/")[-1]
    no_replace = string.replace("_", "/")
    new_path = path.replace("_", "/")
    new_path = new_path.replace(no_replace, string)
    print(new_path)

    new_dir = new_path.replace(filename, "")
    print(new_dir)
    print("File saved to: " + new_dir)
    os.system("mv " + path + " " + new_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="/home/siyuan/Downloads/", help="Path to zip files")
    # parser.add_argument('--save_dir', default="/home/siyuan/workspace/CVbyDL/train", help='Path to save DSEC dataset directory')
    parser.add_argument("--string", default="zurich_city", help="String of the data set")
    args = parser.parse_args()

    file_path_list = glob.glob(os.path.join(args.zip_dir, args.string + "*"), recursive=True)
    print(file_path_list)

    for file in file_path_list:
        if "zip" in file:
            unzip_files(file, args.string)

        else:
            copy_files(file, args.string)
