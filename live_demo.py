import pyrealsense2 as rs
from PIL import Image
import cv2
import os
import click

# # from inference.data.mask_mapper import MaskMapper
# from bundlesdf import BundleSdf

import sys
from argparse import ArgumentParser
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

from bundlesdf import *
import numpy as np

def process_depth(depth_image, mask):
    # Apply the mask to the depth image
    assert depth_image.shape == mask.shape, "Depth image and mask must have the same shape"
    masked_depth = depth_image * (mask > 0).astype(depth_image.dtype)
    return masked_depth

@click.command()
@click.option('--name', type=str)
def main(name):
    print("Starting the camera stream...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # breakpoint()
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    out_folder = "/home/yufeiyang/Documents/BundleSDF/live_output"
    live_data = f"/home/yufeiyang/Documents/BundleSDF/live_data/{name}"
    rgb_path = os.path.join(live_data, "rgb")
    os.system(f'rm -rf {rgb_path} && mkdir -p {rgb_path}')
    depth_path = os.path.join(live_data, "depth")
    os.system(f'rm -rf {depth_path} && mkdir -p {depth_path}')
    mask_path = os.path.join(live_data, "masks")
    os.system(f'rm -rf {mask_path} && mkdir -p {mask_path}')
    cam_k_path = os.path.join(live_data, "cam_K.txt")
    # get intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    np.savetxt(cam_k_path, K)

    # K = np.loadtxt(cam_k_path).reshape(3, 3)
    # breakpoint()
    frame_idx = 1
    mask_done = False
    mask = None
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'
    segment_mask = None
    
    # code_dir = "/home/yufeiyang/Documents/BundleSDF"

    print(code_dir)
    os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

    cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
    cfg_bundletrack['SPDLOG'] = 2
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
    use_gui = 4


    # tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=10, use_gui=use_gui)
    print("Bundle sdf loaded")
    ##
    
    try: 
        print("Press ENTER to select mask once. RGB recording will continue afterwards.")
        while True:
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

            elif key == 13 and not mask_done:  # Enter key: only once for mask
                print("Frame captured. Select mask points (click to add points). Press ENTER to finish.")

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

                # Save the current RGB frame with mask
                rgb_filename = os.path.join(rgb_path, f"{frame_idx:05d}.png")
                cv2.imwrite(rgb_filename, color_image)
                print(f"Saved initial RGB frame to {rgb_filename}")
                
                mask_done = True
                processed_depth = process_depth(depth_image, mask)
                cv2.imwrite(os.path.join(depth_path, f"{frame_idx:05d}.png"), processed_depth)


                ####
                segment_mask = np.array(Image.open(os.path.join(mask_path, f"{frame_idx:05d}.png")))
                segment_mask = (segment_mask > 0).astype(np.uint8)
                num_objects = len(np.unique(segment_mask)) - 1
                processor = InferenceCore(network, config=config_file)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                # mapper = MaskMapper()

                frame_idx += 1
                # #########

            elif mask_done:
                # After mask selection, keep saving RGB frames
                rgb_filename = os.path.join(rgb_path, f"{frame_idx:05d}.png")

                # convert numpy array to pytorch tensor format
                frame_torch, _ = image_to_torch(color_image, device=device)
                if frame_idx == 2:
                   # initialize with the mask
                   
                    mask_torch = index_numpy_to_one_hot_torch(segment_mask, num_objects+1).to(device)
                    # the background mask is not fed into the model
                    prediction = processor.step(frame_torch, mask_torch[1:])
                else:
                   # propagate only
                    prediction = processor.step(frame_torch)

                prediction = torch_prob_to_numpy_mask(prediction)
                predicted_mask = prediction.astype(np.uint8) * 255
                cv2.imwrite(os.path.join(mask_path, f"{frame_idx:05d}.png"), predicted_mask)
                cv2.imwrite(rgb_filename, color_image)
                processed_depth = process_depth(depth_image, mask)
                cv2.imwrite(os.path.join(depth_path, f"{frame_idx:05d}.png"), processed_depth)

                # BundleSDF
                color_img = cv2.imread(rgb_filename)
                H0, W0 = color_img.shape[:2]
                depth = cv2.imread(rgb_filename.replace('rgb','depth'),-1)/1e3
                depth_img = cv2.resize(depth, (W0,H0), interpolation=cv2.INTER_NEAREST)
                H, W = depth_img.shape[:2]
                color_img = cv2.resize(color_img, (W,H), interpolation=cv2.INTER_NEAREST)
                depth_img = cv2.resize(depth_img, (W,H), interpolation=cv2.INTER_NEAREST)

                mask = cv2.imread(rgb_filename.replace('rgb','masks'),-1)
                if len(mask.shape)==3:
                    mask = (mask.sum(axis=-1)>0).astype(np.uint8)
                    mask = cv2.resize(mask, (W0,H0), interpolation=cv2.INTER_NEAREST)
                
                if cfg_bundletrack['erode_mask']>0:
                    kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
                    mask = cv2.erode(mask.astype(np.uint8), kernel)
                id_str = os.path.basename(rgb_filename).replace('.png','')
                pose_in_model = np.eye(4)
                # tracker.run(color_img, depth_img, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)
                    # breakpoint()
                ####

                frame_idx += 1
                print("frame_idx:", frame_idx)
        # tracker.on_finish()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
