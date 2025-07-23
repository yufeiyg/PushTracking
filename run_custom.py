# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import datetime
import pyrealsense2 as rs
from bundlesdf import *
import argparse
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter
import meshcat
import meshcat.geometry as g
import pymeshfix
import nvdiffrast.torch as dr


sys.path.append("/home/yufeiyang/Documents/FoundationPose")
from mask import *
from estimater import *
from datareader import *
code_dir = os.path.dirname(os.path.realpath(__file__))

def run_one_video(video_dir='/home/bowen/debug/2022-11-18-15-10-24_milk', out_folder='/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk/', use_segmenter=False, use_gui=False):
  set_seed(0)

  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(args.debug_level)
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = 3
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5
  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  if use_segmenter:
    segmenter = Segmenter()

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=use_gui)

  reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)


  for i in range(0,len(reader.color_files),args.stride):
    color_file = reader.color_files[i]
    color = cv2.imread(color_file)
    H0, W0 = color.shape[:2]
    depth = reader.get_depth(i)
    H,W = depth.shape[:2]
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)

    if i==0:
      mask = reader.get_mask(0)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
      if use_segmenter:
        mask = segmenter.run(color_file.replace('rgb','masks'))
    else:
      if use_segmenter:
        mask = segmenter.run(color_file.replace('rgb','masks'))
      else:
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

    if cfg_bundletrack['erode_mask']>0:
      kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
      mask = cv2.erode(mask.astype(np.uint8), kernel)

    id_str = reader.id_strs[i]
    pose_in_model = np.eye(4)

    K = reader.K.copy()

    tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)

  tracker.on_finish()

  run_one_video_global_nerf(out_folder=out_folder)



def run_one_video_global_nerf(out_folder='/home/bowen/debug/bundlesdf_scan_coffee_415'):
  set_seed(0)

  out_folder += '/'   #!NOTE there has to be a / in the end

  cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml",'r'))
  cfg_bundletrack['debug_dir'] = out_folder
  cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml",'r'))
  cfg_nerf['n_step'] = 2000
  cfg_nerf['N_samples'] = 64
  cfg_nerf['N_samples_around_depth'] = 256
  cfg_nerf['first_frame_weight'] = 1
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['finest_res'] = 256
  cfg_nerf['num_levels'] = 16
  cfg_nerf['mesh_resolution'] = 0.002
  cfg_nerf['n_train_image'] = 500
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['frame_features'] = 2
  cfg_nerf['rgb_weight'] = 100

  cfg_nerf['i_img'] = np.inf
  cfg_nerf['i_mesh'] = cfg_nerf['i_img']
  cfg_nerf['i_nerf_normals'] = cfg_nerf['i_img']
  cfg_nerf['i_save_ray'] = cfg_nerf['i_img']

  cfg_nerf['datadir'] = f"{out_folder}/nerf_with_bundletrack_online"
  cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])

  os.makedirs(cfg_nerf['datadir'],exist_ok=True)

  cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  reader = YcbineoatReader(video_dir=f"{args.video_dir}/{args.object_name}", downscale=1)

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)
  tracker.cfg_nerf = cfg_nerf
  tracker.run_global_nerf(reader=reader, get_texture=True, tex_res=512)
  tracker.on_finish()

  print(f"Done")


def postprocess_mesh(out_folder):
  mesh_files = sorted(glob.glob(f'{out_folder}/**/nerf/*normalized_space.obj',recursive=True))
  print(f"Using {mesh_files[-1]}")
  os.makedirs(f"{out_folder}/mesh/",exist_ok=True)

  print(f"\nSaving meshes to {out_folder}/mesh/\n")

  mesh = trimesh.load(mesh_files[-1])
  with open(f'{os.path.dirname(mesh_files[-1])}/config.yml','r') as ff:
    cfg = yaml.load(ff)
  tf = np.eye(4)
  tf[:3,3] = cfg['translation']
  tf1 = np.eye(4)
  tf1[:3,:3] *= cfg['sc_factor']
  tf = tf1@tf
  mesh.apply_transform(np.linalg.inv(tf))
  mesh.export(f"{out_folder}/mesh/mesh_real_scale.obj")

  components = trimesh_split(mesh, min_edge=1000)
  best_component = None
  best_size = 0
  for component in components:
    dists = np.linalg.norm(component.vertices,axis=-1)
    if len(component.vertices)>best_size:
      best_size = len(component.vertices)
      best_component = component
  mesh = trimesh_clean(best_component)

  mesh.export(f"{out_folder}/mesh/mesh_biggest_component.obj")
  mesh = trimesh.smoothing.filter_laplacian(mesh,lamb=0.5, iterations=3, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)
  mesh.export(f'{out_folder}/mesh/mesh_biggest_component_smoothed.obj')



def draw_pose():
  K = np.loadtxt(f'{args.out_folder}/cam_K.txt').reshape(3,3)
  color_files = sorted(glob.glob(f'{args.out_folder}/color/*'))
  mesh = trimesh.load(f'{args.out_folder}/textured_mesh.obj')
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  out_dir = f'{args.out_folder}/pose_vis'
  os.makedirs(out_dir, exist_ok=True)
  logging.info(f"Saving to {out_dir}")
  for color_file in color_files:
    color = imageio.imread(color_file)
    pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
    pose = pose@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
    id_str = os.path.basename(color_file).replace('.png','')
    imageio.imwrite(f'{out_dir}/{id_str}.png', vis)

def numerical_sort(value):
    # Extract the first number found in the filename
    match = re.search(r'(\d+)', os.path.basename(value))
    return int(match.group(1)) if match else -1

def load_matrix_from_txt(path):
    data = np.loadtxt(path)
    if data.size != 16:
        raise ValueError(f"File {path} does not contain a 4x4 matrix.")
    return data.reshape(4, 4)

def rotate_fill_mesh(out_folder, world_T_cam, obj_name):
  raw_mesh_path = f'{out_folder}/textured_mesh.obj'
  pose_folder = f'{out_folder}/ob_in_cam'
  pose_files = sorted(glob.glob(os.path.join(pose_folder, "*.txt")), key=numerical_sort)


  # Load all poses first
  poses = [load_matrix_from_txt(pf) for pf in pose_files]
  last_frame_pose = poses[-1]

  scene_or_mesh = trimesh.load(raw_mesh_path)
  # Load the camera extrinsics
  trimesh_mesh = scene_or_mesh
  world_T_object = world_T_cam @ last_frame_pose
  print(world_T_object)
  # === 2. Create a MeshCat visualizer ===
  vis = meshcat.Visualizer().open()
  vis.delete()  # Clear the scene

  trimesh_mesh.apply_transform(world_T_object)
  trimesh_mesh.apply_translation(-trimesh_mesh.centroid)

  # Create a MeshCat mesh object from Trimesh geometry
  vertices = trimesh_mesh.vertices.astype(np.float32)
  faces = trimesh_mesh.faces.astype(np.uint32)
  meshcat_mesh = g.TriangularMeshGeometry(vertices, faces)

  # test watertight
  if not trimesh_mesh.is_watertight:
      print("Raw mesh from BundleSDF is not watertight")
      trimesh_mesh.merge_vertices()
      trimesh_mesh.remove_duplicate_faces()
      trimesh_mesh.remove_degenerate_faces()
      trimesh_mesh.remove_unreferenced_vertices()
      meshfix = pymeshfix.MeshFix(trimesh_mesh.vertices, trimesh_mesh.faces)
      meshfix.repair()

      fixed_mesh = trimesh.Trimesh(meshfix.v, meshfix.f)
  print("Is the mesh watertight?", fixed_mesh.is_watertight)
  # Set the object in MeshCat
  vis["object"].set_object(meshcat_mesh, g.MeshLambertMaterial(color=0x00FF00))
  vis["object"].set_transform(np.eye(4))
  # export the mesh
  fixed_mesh.export(f'{obj_name}.obj')

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



if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
  parser.add_argument('--out_folder', type=str, default="/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk")
  parser.add_argument('--use_segmenter', type=int, default=0)
  parser.add_argument('--use_gui', type=int, default=1)
  parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
  parser.add_argument('--debug_level', type=int, default=1, help='higher means more logging')
  parser.add_argument('--object_name', type=str, help='object name for Foundation Pose')
  args = parser.parse_args()
  world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
                          [-0.99248708, 0.11664051, -0.03693756, 0.],
                          [-0.06717635, -0.77182713, -0.63227385, 0.35],
                          [0., 0., 0., 1.]])
  vid_dir = f'{args.video_dir}/{args.object_name}'
  out_dir = f'{args.out_folder}/{args.object_name}'
  cam_k = np.loadtxt(f'{vid_dir}/cam_K.txt').reshape(3,3)
  # run_one_video(video_dir=vid_dir, out_folder=out_dir, use_segmenter=args.use_segmenter, use_gui=args.use_gui)
  # rotate_fill_mesh(out_folder=out_dir, world_T_cam=world_T_cam, obj_name=args.object_name)

  # Foundation Pose
  tracking(world_T_cam, cam_k, args.object_name)

