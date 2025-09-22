import pyrealsense2 as rs
import os
from datetime import datetime
import time
import numpy as np
import argparse
import sys
from multiprocessing import shared_memory, Lock, Process, Manager
import multiprocessing
from lcm_sys.franka_subscriber import controller_subscriber

sys.path.append("/home/yufeiyang/Documents/XMem")

import torch
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask

torch.cuda.empty_cache()

config_file = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
    'num_objects': 1,
    }

torch.autograd.set_grad_enabled(False)

model = "/home/yufeiyang/Documents/BundleSDF/BundleTrack/XMem/saves/XMem-s012.pth"
# Load our checkpoint
network = XMem(config_file, model).cuda().eval()


sys.path.append("/home/yufeiyang/Documents/FoundationPose")
from mask import *
from lcm_systems.pose_publisher import PosePublisher
from estimater import *
from datareader import *
import nvdiffrast.torch as dr
import trimesh
import logging
import cv2
import zmq
import pickle
import socket

code_dir = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
  est_device = 'cuda'
else:
  est_device = 'cpu'

Z_OFFSET = 0.05
Z_ORG = 0.01

def check_downward(pose, cam_K):
  # checking whether z is down
  def project_3d_to_2d(pt,K,ob_in_cam):
    pt = pt.reshape(4,1)
    projected = K @ ((ob_in_cam@pt)[:3,:])
    projected = projected.reshape(-1)
    projected = projected/projected[2]
    # breakpoint()
    return projected.reshape(-1)[:2].round().astype(int)

  zz = np.array([0,0,1,1]).astype(float)
  zz[:3] = zz[:3]*0.1
  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), cam_K, pose))
  zz = tuple(project_3d_to_2d(zz, cam_K, pose))
  if zz[1] > origin[1]:
    return True
  return False

def is_flipped_90(prev_pose, flipped_pose):
    def signed_angle_between_axes(T1, T2, axis, ref_axis):
        """
        Compute signed angle (radians) between axis `axis` of T1 and T2,
        using `ref_axis` (3D vector) as the reference for sign.
        """
        R1, R2 = T1[:3,:3], T2[:3,:3]
        u = R1[:,axis] / np.linalg.norm(R1[:,axis])
        v = R2[:,axis] / np.linalg.norm(R2[:,axis])
        ref = ref_axis / np.linalg.norm(ref_axis)

        cross = np.cross(u, v)
        dot = np.dot(u, v)
        signed_angle = np.arctan2(np.dot(ref, cross), dot)
        return signed_angle
    tol=np.deg2rad(10)
    # flipped_angle = 0
    x_diff = signed_angle_between_axes(prev_pose, flipped_pose, 0, [0, 0, 1])
    y_diff = signed_angle_between_axes(prev_pose, flipped_pose, 1, [0, 0, 1])
    if abs(abs(x_diff) - np.pi/2) < tol or abs(abs(y_diff) - np.pi/2) < tol:
        theta = -np.sign(x_diff) * np.pi/2   # opposite of detected flip
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        corrected = np.eye(4)
        corrected[:3,:3] = Rz @ flipped_pose[:3,:3]
        corrected[:3,3]  = flipped_pose[:3,3]  # keep same translation
        return corrected, True   # return corrected pose, and flag
    return flipped_pose, False

def check_orientation(T, world_T_cam):
    """
    Check which axis of T is pointing most vertical (closest to world ±Z).
    Returns a string like 'x up', 'y down', 'z up', etc.
    """
    T = world_T_cam @ T
    R = T[:3, :3]
    world_z = np.array([0, 0, 1])

    axes = {
        "x": R[:, 0],
        "y": R[:, 1],
        "z": R[:, 2],
    }

    best_axis, best_val = None, 0
    for name, axis in axes.items():
        dot = np.dot(axis, world_z)
        if abs(dot) > abs(best_val):
            best_axis, best_val = name, dot

    if best_val >= 0:
        return f"{best_axis} up"
    else:
        return f"{best_axis} down"


def is_consistent(prev_pose, flipped_pose):
    def angle_between_axes(T1, T2, dir):
        R1 = T1[:3,:3]
        R2 = T2[:3,:3]
        x1 = R1[:,dir] / np.linalg.norm(R1[:,dir])
        x2 = R2[:,dir] / np.linalg.norm(R2[:,dir])
        dot = np.clip(np.dot(x1, x2), -1.0, 1.0)
        return np.arccos(dot)  # radians
    tol=np.deg2rad(20)
    x_diff = angle_between_axes(prev_pose, flipped_pose, 0)
    y_diff = angle_between_axes(prev_pose, flipped_pose, 1)
    # if x_diff > np.pi / 3 or y_diff > np.pi / 3:
    #     return False
    if abs(x_diff - np.pi) < tol or abs(y_diff - np.pi) < tol:
        return False  # inconsistent (flipped)
    return True


# Shared memory names (choose unique names if you run multiple cameras)
COLOR_SHM_NAME = "realsense_color_shm_v1"
DEPTH_SHM_NAME = "realsense_depth_shm_v1"
META_NAME = "realsense_meta"  # Manager Namespace, not raw shm
depth_scale = 0.0010000000474974513
MASK_GAP = 6
Rx_180 = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float32)
Ry_180 = np.array([
    [-1,  0,  0, 0],
    [0,  1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])
Rz_180 = np.array([
    [-1,  0,  0, 0],
    [0,  -1,  0, 0],
    [0,  0, 1, 0],
    [0,  0,  0, 1]
])
Rx_90 = np.array([
    [1,  0,  0, 0],
    [0,  0, -1, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1]
], dtype=np.float32)
Ry_90 = np.array([
    [0,  0,  1, 0],
    [0,  1,  0, 0],
    [-1, 0,  0, 0],
    [0,  0,  0, 1]
], dtype=np.float32)
Rx_minus_90 = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1]
], dtype=np.float32)
Ry_minus_90 = np.array([
    [0,  0, -1, 0],
    [0,  1,  0, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1]
], dtype=np.float32)

def tracking(world_T_cam, cam_K, obj_name):
    register_status_path = f"{code_dir}/assets/register.txt"
    num_frame = 60
    re_register_freq = num_frame * 18
    controller_listener = controller_subscriber()
    mesh_file = f"{code_dir}/assets_textured/{obj_name}.obj"
    mesh = trimesh.load(mesh_file, force='mesh')
    debug = 1
    est_refine_iter = 5
    debug_dir = f"{code_dir}/foundationPose/{obj_name}"
    track_refine_iter = 2
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam {debug_dir}/masks')
    mask_path = os.path.join(debug_dir, "masks")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    z_height = extents[0]
    mesh_T = mesh.bounding_box_oriented.primitive.transform
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug,
    glctx=glctx,
    hardcoded_initial_rot_mat=None,
    )
    logging.info("estimator initialization done")
    try:
        color_shm = shared_memory.SharedMemory(name=COLOR_SHM_NAME)
        depth_shm = shared_memory.SharedMemory(name=DEPTH_SHM_NAME)
    except FileNotFoundError:
        print("Shared memory blocks not found. Run producer first.")
        return
    width = 640
    height = 480
    channels = 3

    color_buf = np.ndarray((height, width, channels), dtype=np.uint8, buffer=color_shm.buf)
    depth_buf = np.ndarray((height, width), dtype=np.uint16, buffer=depth_shm.buf)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    i = 0
    lcm_pose_publisher = PosePublisher(obj_name)
    Estimating = True
    keep_gui_window_open = True
    # time.sleep(3)
    prev_pose = None

    previous_controller_mode = False

    ready_to_register = False
    try:
        while Estimating:
            start_time = time.perf_counter()
            controller_listener.run()
            is_c3 = controller_listener.get_controller_mode()  # True if in C3
            print("current controller mode", is_c3)
            ########
            color_image = color_buf.copy()
            depth_image = depth_buf.copy()/1e3
            if i == 0:
                create_mask(color_image, obj_name)
                mask = cv2.imread(f'{code_dir}/assets/mask_{obj_name}.png')
                # Initialize Xmem
                s_mask = np.array(mask)
                segment_mask = (mask > 0).astype(np.uint8)
                num_objects = len(np.unique(segment_mask)) - 1
                processor = InferenceCore(network, config=config_file)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                segment_mask = segment_mask[:, :, 0]
            # Scale depth image to mm
            depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
            if cv2.waitKey(1) == 13:
                Estimating = False
                break   
            
            logging.info(f'i:{i}')
            H, W = cv2.resize(color_image, (640,480)).shape[:2]
            color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
            
            depth[(depth<0.1) | (depth>=np.inf)] = 0
            frame_torch, _ = image_to_torch(color_image, device=est_device)
            with open(register_status_path, "r") as f:
                resgiter_status = f.read()
            if i == 0:
                if len(mask.shape)==3:
                    for c in range(3):
                        if mask[...,c].sum()>0:
                            mask = mask[...,c]
                            break
                mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask,
                                    iteration=est_refine_iter)
                mask_torch = index_numpy_to_one_hot_torch(segment_mask, num_objects+1).to(est_device)
                prediction = processor.step(frame_torch, mask_torch[1:])
                with open(register_status_path, "w") as f:
                    f.write("False")

            else:
                pose = est.track_one(rgb=color, depth=depth, K=cam_K,
                                        iteration=track_refine_iter)


            cam_to_object = pose.copy()
            obj_pose_in_world = world_T_cam @ cam_to_object
            obj_pose_in_world[2, 3] = z_height/2 -0.022
        
            lcm_pose_publisher.publish_pose(obj_name, obj_pose_in_world)
            center_pose = pose@np.linalg.inv(to_origin)
            if i % re_register_freq == 0:
                ready_to_register = True

            if keep_gui_window_open:
                vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow("debug", vis[...,::-1])
                key = cv2.waitKey(1)

                if debug <= 1 and keep_gui_window_open and (key==ord("q")):
                    cv2.destroyWindow("debug")
                    # cv2.destroyWindow(f"mask_{obj_name}")
                    keep_gui_window_open = False
            i += 1
            previous_controller_mode = is_c3
    finally:
        print("Tracking finished")
        # color_shm.close()
        # depth_shm.close()

def get_transform(base_path):
    # check if this is a valid path
    if os.path.exists(base_path):
        print("Path exists.")
    else:
        raise NotADirectoryError(f"Path is not a directory: {base_path}")
    folders = [
        f for f in os.listdir(base_path)
        # if os.path.isdir(os.path.join(base_path, f))
        # and f[:19].count('-') == 5 and '_' in f
    ]
    # Parse folder names as datetime objects
    folders_with_dates = []
    for folder in folders:
        try:
            dt = datetime.datetime.strptime(folder[:19], "%Y-%m-%d_%H-%M-%S")
            folders_with_dates.append((dt, folder))
        except ValueError:
            continue

    # Find the newest one
    if folders_with_dates:
        newest = max(folders_with_dates)[1]
        print("Newest folder:", newest)
    else:
        print("No valid timestamp folders found.")
    calibration_mat = f'{base_path}/{newest}/color_tf_world.npy'
    world_T_cam = np.load(calibration_mat)
    return np.linalg.inv(world_T_cam)


if __name__ == "__main__":
    world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
    parser.add_argument('--object_name', type=str, help='object name for Foundation Pose')
    args = parser.parse_args()
    video_dir = f"{code_dir}/live_data"
    vid_dir = f'{video_dir}/{args.object_name}'
    cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
    tracking(world_T_cam, cam_k, args.object_name)
  
    # consumer_main()