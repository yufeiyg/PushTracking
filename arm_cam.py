import numpy as np
import pyrealsense2 as rs
import cv2
import sys
import lcm
from lcm_sys.lcm_subscriber import FrankaJointSubscriber
import click
import os
from pydrake.all import (
    MultibodyPlant, Parser, RigidTransform
)
from pydrake.common import FindResourceOrThrow
code_dir = os.path.dirname(os.path.realpath(__file__))

def process_depth(depth_image, mask):
    # Apply the mask to the depth image
    assert depth_image.shape == mask.shape, "Depth image and mask must have the same shape"
    masked_depth = depth_image * (mask > 0).astype(depth_image.dtype)
    return masked_depth

def get_serial_num():
    ctx = rs.context()
    connected_devices = ctx.query_devices()
    for dev in connected_devices:
        print("dev name:", dev.get_info(rs.camera_info.name))
        print("serial number:", dev.get_info(rs.camera_info.serial_number))

def collect_data(name):
    # 037522250177 is tracking cam; 341222300913 is arm cam
    arm_serial = "341222300913"
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(arm_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])

    # specify the save directory
    base_folder = f'{code_dir}/arm_data/{name}'
    rgb_path = os.path.join(base_folder, "rgb")
    os.system(f'rm -rf {rgb_path} && mkdir -p {rgb_path}')
    depth_path = os.path.join(base_folder, "depth")
    os.system(f'rm -rf {depth_path} && mkdir -p {depth_path}')
    mask_path = os.path.join(base_folder, "mask")
    os.system(f'rm -rf {mask_path} && mkdir -p {mask_path}')
    cam_k_path = os.path.join(base_folder, "cam_K.txt")
    np.savetxt(cam_k_path, K)
    joint_path = cam_k_path = os.path.join(base_folder, "joint_config.npy")
    # Joint position subcriber
    franka_listener = FrankaJointSubscriber()
    # breakpoint()
    joint_positions = []

    mask_done = False
    frame_idx = 1
    try:
        while True:
            # setting up the camera
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imshow("RGB Stream", color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # listen once
            franka_listener.run()
            fk_joint_pos = franka_listener.get_joint_pos()

            if key == 13:
                # Press enter to save the image and record joint position
                image_display = color_image.copy()
                points = []

                def select_points(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
                        cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
                        cv2.imshow("Select Mask", image_display)

                cv2.namedWindow("Select Mask")
                cv2.setMouseCallback("Select Mask", select_points)

                while True:
                    cv2.imshow("Select Mask", image_display)
                    mask_key = cv2.waitKey(1) & 0xFF
                    if mask_key == 13:
                        break

                cv2.destroyWindow("Select Mask")

                # Create and save the mask
                mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
                if points:
                    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
                cv2.imwrite(os.path.join(mask_path, f"{frame_idx:05d}.png"), mask)
                # segmenter = Segmenter(mask)
                print(f"Saved mask to {mask_path}")

                joint_positions.append(fk_joint_pos)
                print("Saved joint position", fk_joint_pos)

                # Save the current RGB frame with mask
                rgb_filename = os.path.join(rgb_path, f"{frame_idx:05d}.png")
                cv2.imwrite(rgb_filename, color_image)
                print(f"Saved initial RGB frame to {rgb_filename}")

                processed_depth = process_depth(depth_image, mask)
                depth_filename = os.path.join(depth_path, f"{frame_idx:05d}.png")
                cv2.imwrite(depth_filename, processed_depth)
                print(f'Saved masked depth to {depth_filename}')

                frame_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # save the joint_info
    joint_positions = np.array(joint_positions)
    np.save(joint_path, joint_positions)

def get_transform(name):
    joint_folder = f'{code_dir}/arm_data/{name}/joint_config.npy'
    joint_angle = np.load(joint_folder)  # 5 by 7

    # Ue pydrake forward kinematics to get the end effector position from joint positions
    # Create a plant and load your robot

    urdf_path = FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/franka_panda.urdf"
    )
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.AddModelFromFile(urdf_path)  # or .sdf
    plant.Finalize()

    # Create a context
    context = plant.CreateDefaultContext()

    # Set joint positions (example: 5-DOF arm)
    q = joint_angle[0]
    plant.SetPositions(context, q)

    # Get frame of the end effector
    end_effector_frame = plant.GetFrameByName("end_effector_link")  # change name accordingly

    # Compute pose of end effector in world frame
    X_WE = plant.CalcRelativeTransform(
        context,
        frame_A=plant.world_frame(),
        frame_B=end_effector_frame
    )

    # Extract translation (position)
    position = X_WE.translation()
    print("End effector position (world frame):", position)

    # Extract rotation matrix
    rotation_matrix = X_WE.rotation().matrix()
    print("End effector rotation:\n", rotation_matrix)
    return joint_angle

@click.command()
@click.option('--name', type=str)
def main(name):
    # collect_data(name)
    get_transform(name)

if __name__=="__main__":
    main()

