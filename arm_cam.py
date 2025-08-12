import numpy as np
import pyrealsense2 as rs
import cv2
import sys
import lcm
from lcm_sys.lcm_subscriber import FrankaJointSubscriber
import click
import os, glob
from pydrake.all import (
    MultibodyPlant, Parser, RigidTransform
)
from pydrake.multibody.tree import JointIndex
from pydrake.common import FindResourceOrThrow
import matplotlib.pyplot as plt
from cv2 import aruco

code_dir = os.path.dirname(os.path.realpath(__file__))

# TODO measure these
BOARD_T_WORLD = np.array([[0, -1, 0, 0.381],
                         [-1, 0, 0, 0.285], 
                         [0, 0, -1, -0.021],
                         [0, 0, 0, 1]])
WORLD_T_POINT = np.array([[1, 0, 0, 0.07855],
                         [0, 1, 0, 0],
                         [0, 0, 1, -0.0282],
                         [0, 0, 0, 1]])
BOARD_SIDE = [0.021, 0.015]

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

def drake_fk(joint_angle, plant, context):
    # 3. Set the 7 joint positions
    q = np.zeros(plant.num_positions())
    q[0] = 1
    q[-7:] = joint_angle[0]
    plant.SetPositions(context, q)

    ee_body = plant.GetBodyByName("panda_link8")

    # 5. Use EvalBodyPoseInWorld to get the transform
    X_WE = plant.EvalBodyPoseInWorld(context, ee_body)

    # 6. Extract the translation (position) component
    ee_position = X_WE.translation()
    ee_rotation = X_WE.rotation().matrix()
    return ee_position, ee_rotation

def get_transform(name, cam_T_W):
    joint_folder = f'{code_dir}/arm_data/{name}/joint_config.npy'
    joint_angle = np.load(joint_folder)  # 5 by 7
    # Create MultibodyPlant and load the Franka URDF
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")  # adjust path
    plant.Finalize()

    context = plant.CreateDefaultContext()

    # checking ee name
    # for i in range(plant.num_joints()):
    #     joint = plant.get_joint(JointIndex(i))
    #     print(f"Joint {i}: name = {joint.name()}, num_positions = {joint.num_positions()}, position_start = {joint.position_start()}")

    for i in range(joint_angle.shape[0]):
        # print(f"Joint {i}: angle = {joint_angle[i]}")
        ee_pos, ee_rot = drake_fk(joint_angle[i], plant, context)

        print("End effector position in world frame:", ee_pos)
        print("End effector rotation in world frame:", ee_rot)

    """
    First frame: calibration. T(ee_cam) = inv(T(W_ee)) inv(T(cam_W)) T(cam_W) is from calibration
    Following frames: T(W_cam) = T(W_ee)T(ee_cam) T(W_ee) is from FK; T(cam_obj) = inv(T(W_cam))T(W_obj)
    """

def get_camEx(name):
    data_folder = f'{code_dir}/arm_data/{name}'
    rgb = sorted(glob. glob(os.path.join(data_folder, "rgb", "*.png")))
    rgbImg = cv2.imread(rgb[0])

    np_color_image_bgr = np.asanyarray(rgbImg)
    np_color_image = np_color_image_bgr[:, :, ::-1]
    plt.imshow(np_color_image)
    plt.show()

    camera_matrix = np.loadtxt(os.path.join(data_folder, "cam_K.txt"))
    # TODO get the distortion coeffs
    distortion_coefficients = 0
    # Aruco tag definitions.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    board = aruco.CharucoBoard((12,9), 0.03, 0.022, aruco_dict)
    charuco_detector_params = aruco.CharucoParameters()
    charuco_detector_params.cameraMatrix = camera_matrix
    charuco_detector_params.distCoeffs = distortion_coefficients
    charuco_detector = aruco.CharucoDetector(
        board, charucoParams=charuco_detector_params)

    # Get board pose.
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(np_color_image_bgr)
    if len(charuco_corners) == 0:
        raise Exception('No charuco corners detected!')
    obj_points, img_points = board.matchImagePoints(
        charuco_corners, charuco_ids)

    # Get the pose of the camera.
    ret, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, distortion_coefficients)
    if not ret:
        raise Exception('Could not solve PnP!')
    
    # Convert transformation matrix.
    C_R_B, _ = cv2.Rodrigues(rvec)
    C_T_B = np.concatenate((C_R_B, tvec), axis=1)
    C_T_B = np.concatenate((C_T_B, np.array([[0, 0, 0, 1]])), axis=0)

    # Define the board to world transformation.
    B_T_W = BOARD_T_WORLD
    C_T_W = C_T_B @ B_T_W

    # Add a point against the Franka platform on the table surface.
    W_T_P = WORLD_T_POINT
    C_T_P = C_T_W @ W_T_P

    # Debugging plot.
    image_debug_viz = cv2.drawFrameAxes(
        np_color_image_bgr,
        camera_matrix,
        distortion_coefficients,
        C_T_B[:3, :3],
        C_T_B[:3, 3:],
        0.1
    )
    image_debug_viz = cv2.drawFrameAxes(
        image_debug_viz,
        camera_matrix,
        distortion_coefficients,
        C_T_W[:3, :3],
        C_T_W[:3, 3:],
        0.08
    )
    image_debug_viz = cv2.drawFrameAxes(
        image_debug_viz,
        camera_matrix,
        distortion_coefficients,
        C_T_P[:3, :3],
        C_T_P[:3, 3:],
        0.08
    )
    plt.imshow(image_debug_viz[:, :, ::-1])
    plt.show()
    return C_T_W

@click.command()
@click.option('--name', type=str)
def main(name):
    # collect_data(name)
    C_T_W = get_camEx(name)
    get_transform(name, C_T_W)

if __name__=="__main__":
    main()

