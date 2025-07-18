import os
import numpy as np
import cv2
depth_folder = "/home/yufeiyang/Documents/BundleSDF/T_push/depth_orig"
mask_folder = "/home/yufeiyang/Documents/BundleSDF/T_push/masks/video1"
output_folder = "/home/yufeiyang/Documents/BundleSDF/T_push/depth"

# list all depth files and mask files
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
if len(depth_files) != len(mask_files):
    print("Error: The number of depth and mask images do not match.")

for depth_file, mask_file in zip(depth_files, mask_files):
    # read in depth and mask images
    depth = cv2.imread(os.path.join(depth_folder, depth_file), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_UNCHANGED)

    depth = np.asanyarray(depth)
    mask = np.asanyarray(mask)
    if depth.shape != mask.shape:

        print(f"Error: Depth and mask images {depth_file} and {mask_file} do not match in size.")
        continue
    masked_depth = depth * (mask > 0).astype(depth.dtype)
    # save depth
    output_path = os.path.join(output_folder, depth_file)
    cv2.imwrite(output_path, masked_depth)
    print(f"Processed {depth_file} and saved to {output_path}")
