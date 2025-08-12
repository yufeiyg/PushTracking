import time
import argparse
import numpy as np
import cv2
from multiprocessing import shared_memory, Lock, Process, Manager
import multiprocessing
import struct
import sys
import os 
import signal
sys.path.append("/home/anything/workspace/FoundationPose")
from mask import *
from datetime import datetime
from lcm_systems.pose_publisher import PosePublisher
from estimater import *
from datareader import *
import nvdiffrast.torch as dr

code_dir = os.path.dirname(os.path.realpath(__file__))
# Shared memory names (choose unique names if you run multiple cameras)
COLOR_SHM_NAME = "realsense_color_shm_v1"
DEPTH_SHM_NAME = "realsense_depth_shm_v1"
META_NAME = "realsense_meta"  # Manager Namespace, not raw shm
depth_scale = 0.0010000000474974513


def tracking(world_T_cam, cam_K, obj_name):
    mesh_file = f"{obj_name}.obj"
    mesh = trimesh.load(mesh_file, force='mesh')
    debug = 1
    est_refine_iter = 5
    debug_dir = f"{code_dir}/foundationPose/{obj_name}"
    track_refine_iter = 2
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
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
    time.sleep(3)
    first_z = 0.
    try:
        while Estimating:
            start_time = time.perf_counter()
            ########
            color_image = color_buf.copy()
            depth_image = depth_buf.copy()/1e3
            if i == 0:
                create_mask(color_image, obj_name)
                mask = cv2.imread(f'mask_{obj_name}.png')
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
            if i == 0:
                if len(mask.shape)==3:
                    for c in range(3):
                        if mask[...,c].sum()>0:
                            mask = mask[...,c]
                            break
                mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask,
                                    iteration=est_refine_iter)
                
                if debug>=3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth, cam_K)
                    valid = depth>=0.1
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            else:
                pose = est.track_one(rgb=color, depth=depth, K=cam_K,
                                        iteration=track_refine_iter)
            os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))
            print("save to " + f'{debug_dir}/ob_in_cam/{i}.txt')

            cam_to_object = pose
            obj_pose_in_world = world_T_cam @ cam_to_object
            # if i == 0:
            #     first_z = obj_pose_in_world[2, 3]
            # else:
            #     obj_pose_in_world[2, 3] = first_z
            obj_pose_in_world[2, 3] = -0.009
        
            lcm_pose_publisher.publish_pose(obj_name, obj_pose_in_world)
            center_pose = pose@np.linalg.inv(to_origin)
            if keep_gui_window_open:
                vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow("debug", vis[...,::-1])
                key = cv2.waitKey(1)

                if debug <= 1 and keep_gui_window_open and (key==ord("q")):
                    cv2.destroyWindow("debug")
                    keep_gui_window_open = False

            if debug>=2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)
            
            i += 1
            print(f"duration: {time.perf_counter() - start_time}")
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
    # world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
    #                         [-0.99248708, 0.11664051, -0.03693756, 0.],
    #                         [-0.06717635, -0.77182713, -0.63227385, 0.35],
    #                         [0., 0., 0., 1.]])
    world_T_cam = get_transform(base_path='/home/anything/workspace/ci_mpc_utils/calibrations')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
    parser.add_argument('--object_name', type=str, help='object name for Foundation Pose')
    args = parser.parse_args()
    video_dir = f"{code_dir}/live_data/"
    vid_dir = f'{video_dir}/{args.object_name}'
    cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
    scale_x = 640 / 1280
    scale_y = 480 / 800
    cam_k[0, :] *= scale_x
    cam_k[1, :] *= scale_y
    tracking(world_T_cam, cam_k, args.object_name)
  
    # consumer_main()