import pyrealsense2 as rs
import os
from datetime import datetime
import time
import numpy as np
import argparse
import sys
sys.path.append("/home/anything/workspace/FoundationPose")
from mask_multi import *
from lcm_systems.pose_publisher import PosePublisher
from estimater import *
from datareader import *
import nvdiffrast.torch as dr
import trimesh
import logging
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))


def tracking(world_T_cam, cam_K, obj_names):
  all_estimates = []
  all_mask = []
  debug = 1
  est_refine_iter = 5
  track_refine_iter = 2

  # for all objects we want to track
  for obj_name in obj_names:
    obj_name = obj_name[0]
    mesh_file = f"{obj_name}.obj"
    mesh = trimesh.load(mesh_file, force='mesh')
    debug_dir = f"{code_dir}/foundationPose/{obj_name}"
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
    all_estimates.append(est)
    logging.info(f"estimator initialization for {obj_name} done")
    create_mask(obj_name)
    mask = cv2.imread(f'{obj_name}_mask.png')
    all_mask.append(mask)

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

  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
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
  all_publisher = []
  for obj_i in range(len(obj_names)):
    lcm_pose_publisher = PosePublisher(obj_names[obj_i][0])
    # breakpoint()
    all_publisher.append(lcm_pose_publisher)
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

        # doing inference
        all_pose = []
        for obj_num in range(len(obj_names)):
          curr_mask = all_mask[obj_num]
          # breakpoint()
          curr_est = all_estimates[obj_num]
          if i == 0:
              if len(curr_mask.shape)==3:
                for c in range(3):
                  if curr_mask[...,c].sum()>0:
                    curr_mask = curr_mask[...,c]
                    break
              curr_mask = cv2.resize(curr_mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
              pose = curr_est.register(K=cam_K, rgb=color, depth=depth, ob_mask=curr_mask,
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
              # if obj_num == 0:
              pose = curr_est.track_one(rgb=color, depth=depth, K=cam_K,
                                  iteration=track_refine_iter)
          all_pose.append(pose)
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))
        print("save to " + f'{debug_dir}/ob_in_cam/{i}.txt')

        for obj_num in range(len(obj_names)):
          cam_to_object = all_pose[obj_num]
          obj_pose_in_world = world_T_cam @ cam_to_object
          all_publisher[obj_num].publish_pose(obj_names[obj_num][0], obj_pose_in_world)
          center_pose = cam_to_object@np.linalg.inv(to_origin)
          if keep_gui_window_open:
              vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
              vis = draw_xyz_axis(color, ob_in_cam=cam_to_object, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
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
  world_T_cam = np.linalg.inv(np.load(calibration_mat))
  return world_T_cam


def comma_separated_list(value):
    return value.split(',')

if __name__ == "__main__":
    # world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
    #                         [-0.99248708, 0.11664051, -0.03693756, 0.],
    #                         [-0.06717635, -0.77182713, -0.63227385, 0.35],
    #                         [0., 0., 0., 1.]])
    world_T_cam = get_transform(base_path='/home/anything/workspace/ci_mpc_utils/calibrations')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
    parser.add_argument('--object_name', nargs='+', type=comma_separated_list, help='object name for Foundation Pose')
    args = parser.parse_args()
    video_dir = f"{code_dir}/live_data/"
    vid_dir = f'{video_dir}{args.object_name[0][0]}'
    cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
    tracking(world_T_cam, cam_k, args.object_name)