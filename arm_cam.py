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
from pydrake.math import RigidTransform
import matplotlib.pyplot as plt
from cv2 import aruco

code_dir = os.path.dirname(os.path.realpath(__file__))

# TODO measure these
# BOARD_T_WORLD = np.array([[1, 0, 0, -0.021],
#                          [0, -1, 0, -0.012], 
#                          [0, 0, -1, 0],
#                          [0, 0, 0, 1]])
BOARD_T_WORLD = np.array([[1, 0, 0, -0.396],
                         [0, -1, 0, 0.436],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])

BASE_T_W = np.array([[1, 0, 0, 0.075],
                    [0, 1, 0, 0.], 
                    [0, 0, 1, -0.03048],
                    [0, 0, 0, 1]])
W_T_OBJ = np.array([[1, 0, 0, 0.402],
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0],
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
    depth_path = os.path.join(base_folder, "depth_enhanced")
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
    plant_q = np.zeros(plant.num_positions())
    plant_q[:7] = joint_angle
    plant.SetPositions(context, plant_q)
    
    ee_body = plant.GetBodyByName("panda_link8")

    return plant.EvalBodyPoseInWorld(context, ee_body).GetAsMatrix4()

def get_transform(name, ee_T_cam):
    joint_folder = f'{code_dir}/arm_data/{name}/joint_config.npy'
    output_folder = f'{code_dir}/arm_data/{name}/cam_in_ob'
    ee_output_folder = f'{code_dir}/arm_data/{name}/ee_pos'
    os.system(f'rm -rf {output_folder} && mkdir -p {output_folder}')
    os.system(f'rm -rf {ee_output_folder} && mkdir -p {ee_output_folder}')
    joint_angle = np.load(joint_folder)  # 5 by 7
    # Create MultibodyPlant and load the Franka URDF
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    instance = parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")
    instance = instance[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"), RigidTransform.Identity())
    plant.Finalize()
    for body_index in plant.GetBodyIndices(instance):
        body = plant.get_body(body_index)
        print(body.name())        
    context = plant.CreateDefaultContext()

    for i in range(joint_angle.shape[0]):
        base_T_ee = drake_fk(joint_angle[i], plant, context)
        ee_T_base = np.linalg.inv(base_T_ee)
        ee_T_world = ee_T_base @ BASE_T_W  # this is correct
        # np.savetxt(f'{ee_output_folder}/{i+1:05d}.txt', ee_T_world)
        W_T_ee = np.linalg.inv(ee_T_world)  # this is correct
        W_T_cam = W_T_ee @ ee_T_cam
        np.savetxt(f'{ee_output_folder}/{i+1:05d}.txt', W_T_ee)
        cam_T_obj = np.linalg.inv(W_T_cam) @ W_T_OBJ  # cam_T_obj = cam_T_W @ W_T_OBJ
        # save the obj_T_cam (cam_in_ob) in a text file
        np.savetxt(f'{output_folder}/{i+1:05d}.txt', np.linalg.inv(cam_T_obj))

    """
    Following frames: T(W_cam) = T(W_ee)T(ee_cam) T(W_ee) is from FK; T(cam_obj) = inv(T(W_cam))T(W_obj)
    """

def get_camEx(rgb_path, camera_matrix, joint_pos):
    '''
    ee_T_cam = ee_T_world @ world_T_cam = ee_T_base @ inv(cam_T_world)
    ee_T_world = ee_T_base @ base_T_world
    '''
    # data_folder = f'{code_dir}/arm_data/calibration_lsqr'
    # rgb = sorted(glob. glob(os.path.join(data_folder, "rgb", "*.png")))
    rgbImg = cv2.imread(rgb_path)

    np_color_image_bgr = np.asanyarray(rgbImg)
    np_color_image = np_color_image_bgr[:, :, ::-1]
    # plt.imshow(np_color_image)
    # plt.show()

    # camera_matrix = np.loadtxt(os.path.join(data_folder, "cam_K.txt"))
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
    W_T_BASE = np.linalg.inv(BASE_T_W) # world_T_base
    C_T_BASE = C_T_W @ W_T_BASE # camera_T_base

    W_T_O = W_T_OBJ
    C_T_O = C_T_W @ W_T_O
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
        C_T_BASE[:3, :3],
        C_T_BASE[:3, 3:],
        0.08
    )
    image_debug_viz = cv2.drawFrameAxes(
        image_debug_viz,
        camera_matrix,
        distortion_coefficients,
        C_T_O[:3, :3],
        C_T_O[:3, 3:],
        0.08
    )

    plt.imshow(image_debug_viz[:, :, ::-1])
    plt.show()
    
    # C_T_W is cam_T_world
    world_T_cam = np.linalg.inv(C_T_W)
    # joint_pos = np.load(os.path.join(data_folder, "joint_config.npy"))
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"), RigidTransform.Identity())
    plant.Finalize()
    context = plant.CreateDefaultContext()
    # breakpoint()
    base_T_ee = drake_fk(joint_pos, plant, context)
    ee_T_W = np.linalg.inv(base_T_ee) @ BASE_T_W  # ee_T_world = ee_T_base @ base_T_world
    ee_T_cam = ee_T_W @ world_T_cam  # ee_T_cam = ee_T_world @ world_T_cam
 
    return ee_T_cam

def camera_calibration():
    data_folder = f'{code_dir}/arm_data/calibration_lsqr'
    rgb = sorted(glob. glob(os.path.join(data_folder, "rgb", "*.png")))
    joint_pos = np.load(os.path.join(data_folder, "joint_config.npy"))
    camera_matrix = np.loadtxt(os.path.join(data_folder, "cam_K.txt"))
    all_eeTcam = []
    for i in range(len(rgb)):
        ee_T_cam = get_camEx(rgb[i], camera_matrix, joint_pos[i])
        all_eeTcam.append(ee_T_cam)
    # run least squares optimization on all_eeTcam to get the most accurate transformation
    matrices = np.array(all_eeTcam)  # (N, 4, 4)
    assert matrices.shape[1:] == (4, 4), "Each matrix must be 4x4"

    # --- Average translation ---
    translations = matrices[:, :3, 3]
    t_avg = np.mean(translations, axis=0)

    # --- Average rotation (using SVD projection) ---
    R_stack = matrices[:, :3, :3]
    R_sum = np.sum(R_stack, axis=0)

    # Project back to SO(3) with SVD
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:  # fix improper rotation
        U[:, -1] *= -1
        R_avg = U @ Vt

    # --- Rebuild transform ---
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    # M = np.stack(all_eeTcam) 
    # ee_T_cam = np.mean(M, axis=0)
    return T_avg

@click.command()
@click.option('--name', type=str)
def main(name):
    # collect_data(name)
    # First frame: calibration. T(ee_cam) = inv(T(W_ee)) inv(T(cam_W)) T(cam_W) is from calibration
    # ee_T_cam = get_camEx()
#     [[ 0.69766779 -0.00837756  0.71615637  0.04929139]
#  [ 0.71616443  0.02286852 -0.69732234 -0.06149469]
#  [-0.01057131  0.99959846  0.02187438  0.03163664]
#  [ 0.          0.          0.          1.        ]]
    # ee_T_cam = camera_calibration()
    # print(ee_T_cam)
    # ee_T_cam = np.array([[0.69766779, -0.00837756,  0.71615637,  0.04929139],
    #                      [0.71616443,  0.02286852, -0.69732234, -0.06149469],
    #                      [-0.01057131,  0.99959846,  0.02187438,  0.03163664],
    #                      [0, 0, 0, 1]])
    ee_T_cam = np.array([[0.69775333, -0.00833757,  0.71628959,  0.04929139],
                         [0.71626033,  0.02285499, -0.69745881, -0.06149469],
                         [-0.01055568,  0.99970402,  0.02191901,  0.03163664],
                         [0, 0, 0, 1]])
    get_transform(name, ee_T_cam)

if __name__=="__main__":
    main()

