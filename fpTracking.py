import pyrealsense2 as rs
import os
import time
import numpy as np
import argparse
import sys
sys.path.append("/home/yufeiyang/Documents/FoundationPose")
from mask import *
from lcm_systems.pose_publisher import PosePublisher
from estimater import *
from datareader import *
import nvdiffrast.torch as dr
import trimesh
import logging
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))


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

  create_mask()
  mask = cv2.imread('mask.png')

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

  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
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
        lcm_pose_publisher.publish_pose(obj_name, obj_pose_in_world)
    
        if keep_gui_window_open:
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
      pipeline.stop()

if __name__ == "__main__":
    world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
                            [-0.99248708, 0.11664051, -0.03693756, 0.],
                            [-0.06717635, -0.77182713, -0.63227385, 0.35],
                            [0., 0., 0., 1.]])
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
    parser.add_argument('--object_name', type=str, help='object name for Foundation Pose')
    args = parser.parse_args()
    vid_dir = f'{args.video_dir}/{args.object_name}'
    cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
    tracking(world_T_cam, cam_k, args.object_name)