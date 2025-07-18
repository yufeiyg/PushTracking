import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import os

def get_mask(rgb_path, depth_path):
    rgb_files = sorted(os.listdir(rgb_path))
    depth_files = sorted(os.listdir(depth_path))

    if len(rgb_files) != len(depth_files):
        print("Error: The number of RGB and depth images do not match.")
        return

    for i in range(len(rgb_files)):
        rgb_file = os.path.join(rgb_path, rgb_files[i])
        depth_file = os.path.join(depth_path, depth_files[i])

        color_image = cv2.imread(rgb_file)
        depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

        # run XMem on the rgb file
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_video', type=int, default=1)
    args = parser.parse_args()
    if args.use_video == 1:  # use video
        video_folder = "/home/yufeiyang/Documents/test"
        rgb_file = f"{video_folder}/realsense_rgb.mp4"
        cap_rgb = cv2.VideoCapture(rgb_file)
        # output_folder = "/home/yufeiyang/Documents/BundleSDF/T_push"
        output_folder = "/home/yufeiyang/Documents/BundleSDF/T_data"
        if not cap_rgb.isOpened():
            print("Error: Could not open video files.")
            exit()
        frame_idx = 1
        while True:
            ret_rgb, color_frame = cap_rgb.read()
            # ret_depth, depth_frame = cap_depth.read()
            if not ret_rgb:
                print("Error: Could not read frame from video.")
                break
            # rgb_output = f"{output_folder}/JPEGImages/video1"
            rgb_output = f"{output_folder}/rgb"

            rgb_path = os.path.join(rgb_output, f"{frame_idx:05d}.png")


            cv2.imwrite(rgb_path, color_frame)
            # cv2.imwrite(depth_path, depth_frame)
            frame_idx += 1

            # Convert images to numpy arrays
            color_image = np.array(color_frame)
            # depth_image = np.array(depth_frame)
            # Display the images
            cv2.imshow("Color Image", color_image)
            # cv2.imshow("Depth Image", depth_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap_rgb.release()
        # cap_depth.release()
    elif args.use_video == 2:  # use image frames
        # read in rgb and depth images from a folder
        rgb_folder = "/home/yufeiyang/Documents/BundleSDF/T_push/JPEGImages/video1"
        depth_folder = "/home/yufeiyang/Documents/BundleSDF/T_push/depth"
        get_mask(rgb_folder, depth_folder)

    # Create a pipeline
    # pipeline = rs.pipeline()

    # # Configure the pipeline to stream color and depth
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # # Start streaming
    # pipeline.start(config)

    # try:
    #     while True:
    #         # Wait for a coherent pair of frames: depth and color
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()
    #         depth_frame = frames.get_depth_frame()

    #         if not color_frame or not depth_frame:
    #             continue

    #         # Convert images to numpy arrays
    #         color_image = np.asanyarray(color_frame.get_data())
    #         depth_image = np.asanyarray(depth_frame.get_data())

    #         # Display the images (you can use cv2.imshow here if needed)
    #         print("Color image shape:", color_image.shape)
    #         print("Depth image shape:", depth_image.shape)

    # finally:
    #     # Stop streaming
    #     pipeline.stop()