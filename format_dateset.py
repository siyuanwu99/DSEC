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
import re


def unzip_files(path, directory, string="zurich_city"):
    """
    unzip files and save to new directory with format
    """
   
    filename = path.replace(".zip", "").split("/")[-1]
    old_dir = path.replace(".zip", "").replace(filename, "")
    
    # match "zurich_city_11_a"    
    z = re.match(string + "_\d+_\w", filename)
    print(z.group())
    new_dir = z.group()
    rest = filename.replace(new_dir, "").replace("_", "/")
    
    save_dir = directory + new_dir + rest

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.system("unzip " + path + " -d " + save_dir)
    print("File unzipped and saved to: " + save_dir)


def copy_files(path, directory, string="zurich_city"):

    filename = path.split("/")[-1]
    old_dir = path.replace(filename, "")
    
    # match "zurich_city_11_a"    
    z = re.match(string + "_\d+_\w", filename)
    print(z.group())
    new_dir = z.group()
    rest = filename.replace(new_dir, "").replace("_", "/")
    
    save_dir = directory + new_dir + rest
    
    os.system("cp " + path + " " + save_dir)
    print("File saved to: " + save_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="/home/siyuan/Downloads/", help="Path to zip files")
    parser.add_argument('--save_dir', default="/home/siyuan/workspace/CVbyDL/train/", help='Path to save DSEC dataset directory')
    parser.add_argument("--string", default="zurich_city", help="String of the data set")
    args = parser.parse_args()

    file_path_list = glob.glob(os.path.join(args.zip_dir, args.string + "*"), recursive=True)
    print(file_path_list)

    for file in file_path_list:
        print()
        if "zip" in file:
            unzip_files(file, args.save_dir, args.string)

        else:
            copy_files(file, args.save_dir, args.string)
