import pyrealsense2 as rs
import os
from datetime import datetime
import time
import numpy as np
import argparse
import sys


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
print("XMem model loaded")


sys.path.append("/home/yufeiyang/Documents/FoundationPose")
from mask_multi import *
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


def tracking(world_T_cam, cam_K, obj_name):
  mesh_file = f"{obj_name}.obj"
  mesh = trimesh.load(mesh_file, force='mesh')
  debug = 1
  est_refine_iter = 5
  debug_dir = f"{code_dir}/foundationPose/{obj_name}"
  track_refine_iter = 2
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam {debug_dir}/masks')
  mask_path = os.path.join(debug_dir, "masks")

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

  create_mask("push_t_white")
  mask = cv2.imread('push_t_white_mask.png')

  # Initialize Xmem
  s_mask = np.array(mask)
  segment_mask = (mask > 0).astype(np.uint8)
  num_objects = len(np.unique(segment_mask)) - 1
  processor = InferenceCore(network, config=config_file)
  processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
  segment_mask = segment_mask[:, :, 0]

  # Create a pipeline
  pipeline = rs.pipeline()

  # Create a config and configure the pipeline to stream
  config = rs.config()

  # Get device product line for setting a supporting resolution
  pipeline_wrapper = rs.pipeline_wrapper(pipeline)
  pipeline_profile = config.resolve(pipeline_wrapper)
  device = pipeline_profile.get_device()
  device_product_line = str(device.get_info(rs.camera_info.product_line))

  found_rgb = False
  for s in device.sensors:
      if s.get_info(rs.camera_info.name) == 'RGB Camera':
          found_rgb = True
          break
  if not found_rgb:
      print("The demo requires Depth camera with Color sensor")
      exit(0)
  num_frame = 30
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, num_frame)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, num_frame)

  re_register_freq = num_frame * 60
  # Start streaming
  profile = pipeline.start(config)

  # Getting the depth sensor's depth scale (see rs-align example for explanation)
  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: " , depth_scale)

  # We will be removing the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance_in_meters = 1 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  # Create an align object
  align_to = rs.stream.color
  align = rs.align(align_to)

  i = 0
  lcm_pose_publisher = PosePublisher(obj_name)
  Estimating = True
  keep_gui_window_open = True
  time.sleep(3)
  try:
     while Estimating:
        start_time = time.perf_counter()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())

        ########
    
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
        if i == 0:
            if len(mask.shape)==3:
              for c in range(3):
                if mask[...,c].sum()>0:
                  mask = mask[...,c]
                  break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=est_refine_iter)
            
            # breakpoint()
            mask_torch = index_numpy_to_one_hot_torch(segment_mask, num_objects+1).to(est_device)
            prediction = processor.step(frame_torch, mask_torch[1:])
        elif i % re_register_freq == 0:
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=predicted_mask,
                                iteration=est_refine_iter)
            prediction = processor.step(frame_torch)
            cv2.imwrite(os.path.join(mask_path, f"{i:05d}.png"), predicted_mask)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K,
                                 iteration=track_refine_iter)
            prediction = processor.step(frame_torch)
        prediction = torch_prob_to_numpy_mask(prediction)
        predicted_mask = prediction.astype(np.uint8) * 255

        
        cv2.imshow("mask", predicted_mask)
        # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        # np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))
        # print("save to " + f'{debug_dir}/ob_in_cam/{i}.txt')
        if check_downward(pose, cam_K):
          #  rotate pose by 180 degrees around the y axis
          Rx_180 = np.array([
              [1,  0,  0, 0],
              [0, -1,  0, 0],
              [0,  0, -1, 0],
              [0,  0,  0, 1]
          ], dtype=np.float32)

          # T_y_180 = np.eye(4)
          # T_y_180[:3, :3] = Rx_180
          pose = pose @ Rx_180

        cam_to_object = pose
        obj_pose_in_world = world_T_cam @ cam_to_object
        obj_pose_in_world[2, 3] = -0.0085
        lcm_pose_publisher.publish_pose(obj_name, obj_pose_in_world)
        center_pose = pose@np.linalg.inv(to_origin)
        if keep_gui_window_open:
            vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow("debug", vis[...,::-1])
            key = cv2.waitKey(1)

            if debug <= 1 and keep_gui_window_open and (key==ord("q")):
              cv2.destroyWindow("debug")
              cv2.destroyWindow("mask")
              keep_gui_window_open = False   
        i += 1
        print(f"duration: {time.perf_counter() - start_time}")
  finally:
      pipeline.stop()

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
    world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
    parser.add_argument('--object_name', type=str, help='object name for Foundation Pose')
    args = parser.parse_args()
    video_dir = f"{code_dir}/live_data/"
    vid_dir = f'{video_dir}/{args.object_name}'
    cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
    tracking(world_T_cam, cam_k, args.object_name)