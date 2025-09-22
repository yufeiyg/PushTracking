import glob
import re
import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

import datetime
def view_depth():
    depth_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/A_shape/depth_enhanced/00001.png"
    # depth_path = "/home/yufeiyang/Documents/FoundationPose/ycbv/ref_views_4/ob_0000001/depth_enhanced/0000000.png"
    # visualize this depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32)  # Convert to meters
    cv2.imshow("Depth Image", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_mesh_offset():
    mesh = trimesh.load('/home/yufeiyang/Documents/BundleSDF/wood_block.obj')
    theta = np.deg2rad(180)  # convert degrees to radians
    Rx_100 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    mesh.apply_transform(Rx_100)
    mesh.export(f'/home/yufeiyang/Documents/BundleSDF/assets/wood_block_flipped.obj')

def draw_transformation(trans, ax, idx):
    # Given cam_in_ob, a 4 by 4 transformation matrix, draw this as a triad with matplotlip
    origin = trans[:3, 3]
    x_axis = trans[:3, 0]
    y_axis = trans[:3, 1]
    z_axis = trans[:3, 2]
    ax.quiver(*origin, *x_axis, color='r', length=0.3)
    ax.quiver(*origin, *y_axis, color='g', length=0.3)
    ax.quiver(*origin, *z_axis, color='b', length=0.3)
    ax.text(*(origin + 0.01), str(idx), color='k', fontsize=10, weight='bold')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    return

def view_transformation(cam_in_ob_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # load all files in cam_in_ob_path
    i = 0
    for file in os.listdir(cam_in_ob_path):
        if file.endswith(".txt"):
            ee_pose = np.loadtxt(os.path.join(cam_in_ob_path, file))
            idx = file.split('.')[0][-2:]
            draw_transformation(ee_pose, ax, idx)
            i += 1
    draw_transformation(np.eye(4), ax, "origin")
    plt.show()

def view_basic_frame(ob_in_cam):
    BASE_T_W = np.array([[1, 0, 0, 0.075],
                        [0, 1, 0, 0.], 
                        [0, 0, 1, -0.03048],
                        [0, 0, 0, 1]])
    W_T_OBJ = np.array([[1, 0, 0, 0.402],
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    OBJ_T_W = np.linalg.inv(W_T_OBJ)
    ee_T_W = np.array([[0.60902451, -0.4516771, 0.65197925, 0.57752245],
                          [0.7258722, 0.64872026, -0.22862978, -0.04328703],
                          [0.31968531, -0.61249475, -0.72294639, 0.24105176],
                          [0., 0., 0., 1.]])

    ee_T_Base = np.array([[-0.4516771, 0.60902451, 0.65197925, 0.63127056],
                           [0.64872026, 0.7258722, -0.22862978, -0.09890968],
                           [-0.61249475, 0.31968531, -0.72294639, 0.26495346],
                           [0., 0., 0., 1.]])

    ee_T_cam = np.array([[ 0.40092747, -0.9040574 , -0.14811277,  0.42326053],
                          [ 0.82771455,  0.42676521, -0.36436256,  0.16227779],
                          [ 0.39261404,  0.02348786,  0.91940336, -0.27654754],
                          [ 0.        ,  0.        ,  0.        ,  1.        ]])

    cam_T_W = np.array([[ 0.1153914 ,  0.97050258, -0.21168272,  0.0949156 ],
                         [ 0.67080707, -0.23330739, -0.70397836, -0.21503224],
                         [-0.73259996, -0.06076521, -0.67794166,  0.5279345 ],
                         [ 0.        ,  0.        ,  0.        ,  1.        ]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_transformation(np.eye(4), ax, "origin")
    # draw_transformation(BASE_T_W, ax, "base_T_world")
    draw_transformation(ob_in_cam, ax, "ob_in_cam")
    # draw_transformation(cam_T_W, ax, "cam_T_W")
    # draw_transformation(np.linalg.inv(ee_T_cam), ax, "ee_T_cam")
    # draw_transformation(ee_T_W, ax, "ee_T_W")
    # draw_transformation(ee_T_Base, ax, "ee_T_Base")

def is_consistent(prev_pose, flipped_pose):
    def angle_between_axes(T1, T2, dir):
        R1 = T1[:3,:3]
        R2 = T2[:3,:3]
        x1 = R1[:,dir] / np.linalg.norm(R1[:,dir])
        x2 = R2[:,dir] / np.linalg.norm(R2[:,dir])
        dot = np.clip(np.dot(x1, x2), -1.0, 1.0)
        return np.arccos(dot)  # radians
    tol=np.deg2rad(30)
    x_diff = angle_between_axes(prev_pose, flipped_pose, 0)
    y_diff = angle_between_axes(prev_pose, flipped_pose, 1)
    if abs(x_diff - np.pi) < tol or abs(y_diff - np.pi) < tol:
        return False  # inconsistent (flipped)
    return True

def debug():
    origin = np.eye(4)
    theta = np.deg2rad(180)  # convert degrees to radians
    Rx_180 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    theta = np.deg2rad(180)
    Ry_180 = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    theta = np.deg2rad(180)  # convert degrees to radians
    Rz_180 = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ], dtype=np.float32)
    world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
    prev_pose = np.array([[ 9.975585937500000000e-01,  1.979980468750000000e-01,  1.178741455078125000e-02,  5.281279981136322021e-02],
                          [ 5.084228515625000000e-02, -2.331542968750000000e-01, -1.026367187500000000e+00,  1.066136956214904785e-01],
                          [-1.790771484375000000e-01,  9.697265625000000000e-01, -2.331542968750000000e-01,  6.643654108047485352e-01],
                          [ 0.000000000000000000e+00,  0.000000000000000000e+00,  0.000000000000000000e+00,  1.000000000000000000e+00]])
    view_basic_frame(prev_pose)
    pose = np.array([[-8.500976562500000000e-01, -5.654296875000000000e-01,  8.445739746093750000e-03,  5.300060659646987915e-02],
                     [-1.176757812500000000e-01,  1.811523437500000000e-01, -1.030273437500000000e+00,  1.074103787541389465e-01],
                     [ 5.415039062500000000e-01, -8.256835937500000000e-01, -2.131347656250000000e-01,  6.665469408035278320e-01],
                     [ 0.000000000000000000e+00,  0.000000000000000000e+00,  0.000000000000000000e+00,  1.000000000000000000e+00]])
    case = check_orientation(pose, world_T_cam)
    # breakpoint()
    if case == "z down":
        flipped_pose = pose @ Rx_180 
        if not is_consistent(prev_pose, flipped_pose):
            flipped_pose = flipped_pose @ Rz_180
        pose = flipped_pose
    # breakpoint()
    case = check_orientation(pose, world_T_cam)
    if not is_consistent(prev_pose, pose) and case == "z up":
        breakpoint()
        pose = pose @ Rz_180
    view_basic_frame(pose)



    # flipped_pose = pose @ Rz_180
    # view_basic_frame(pose)
    # world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
    # # check_orientation(prev_pose, world_T_cam)
    # flipped_pose, is_flipped = is_flipped_90(prev_pose, pose)
    # if is_flipped:
    #     print('flipped 90')
    #     pose = flipped_pose
    # view_basic_frame(pose)
    # check_orientation(pose, world_T_cam)

    plt.show()

def is_flipped_90(prev_pose, flipped_pose):
    def signed_angle_between_axes(T1, T2, axis, ref_axis=2):
        """
        Compute signed angle (radians) between axis `axis` of T1 and T2,
        using `ref_axis` (3D vector) as the reference for sign.
        """
        R1, R2 = T1[:3,:3], T2[:3,:3]
        u = R1[:,axis] / np.linalg.norm(R1[:,axis]).reshape(-1)
        v = R2[:,axis] / np.linalg.norm(R2[:,axis]).reshape(-1)
        # Take local Z of T1, expressed in world frame
        ref = R1[:,ref_axis] / np.linalg.norm(R1[:,ref_axis])

        cross = np.cross(u, v)
        dot = np.dot(u, v)
        signed_angle = np.arctan2(np.dot(ref, cross), dot)
        return signed_angle

    tol=np.deg2rad(35)
    # flipped_angle = 0
    x_diff = signed_angle_between_axes(prev_pose, flipped_pose, 0)
    y_diff = signed_angle_between_axes(prev_pose, flipped_pose, 1)
    breakpoint()
    if abs(x_diff) > np.deg2rad(70) or abs(y_diff) > np.deg2rad(70):
    # if abs(abs(x_diff) - np.pi/2) < tol or abs(abs(y_diff) - np.pi/2) < tol:
        R1 = prev_pose[:3,:3]
        R2 = flipped_pose[:3,:3]
        # Build correction rotation: align local x-axis of flipped to prev
        u = R1[:,0] / np.linalg.norm(R1[:,0])   # prev x-axis
        v = R2[:,0] / np.linalg.norm(R2[:,0])   # flipped x-axis

        # Cross product gives axis to rotate v into u
        cross = np.cross(v, u)
        dot   = np.dot(v, u)

        # Angle between them (should be ≈ ±90°)
        angle = np.arctan2(np.linalg.norm(cross), dot)

        # Project correction axis onto *local z* of flipped_pose
        local_z = R2[:,2]
        sign = np.sign(np.dot(cross, local_z))

        theta = sign * angle

        # Local Z rotation
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        corrected = np.eye(4)
        corrected[:3,:3] = R2 @ Rz    # rotate around local Z
        corrected[:3,3]  = flipped_pose[:3,3]
        return corrected, True   # return corrected pose, and flag
    return flipped_pose, False

def check_orientation(T, world_T_cam):
    """
    Check which axis of T is pointing most vertical (closest to world ±Z).
    Returns a string like 'x up', 'y down', 'z up', etc.
    """
    T = world_T_cam @ T
    # view_basic_frame(T)
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
def numerical_sort(value):
    # Extract the first number found in the filename
    match = re.search(r'(\d+)', os.path.basename(value))
    return int(match.group(1)) if match else -1

def load_matrix_from_txt(path):
    data = np.loadtxt(path)
    if data.size != 16:
        raise ValueError(f"File {path} does not contain a 4x4 matrix.")
    return data.reshape(4, 4)

def rotate_textured_mesh():
    obj_name = "dunkin_box"
    mesh_path = f'/home/yufeiyang/Documents/BundleSDF/assets/{obj_name}.obj'
    output_folder = f'/home/yufeiyang/Documents/BundleSDF/debug_output/{obj_name}'
    pose_folder = f'{output_folder}/ob_in_cam'
    pose_files = sorted(glob.glob(os.path.join(pose_folder, "*.txt")), key=numerical_sort)
    poses = [load_matrix_from_txt(pf) for pf in pose_files]
    last_frame_pose = poses[-1]
    world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
    trimesh_mesh = trimesh.load(mesh_path)
    world_T_object = world_T_cam @ last_frame_pose
    trimesh_mesh.apply_transform(world_T_object)
    trimesh_mesh.apply_translation(-trimesh_mesh.centroid)

    trimesh_mesh.export(f'/home/yufeiyang/Documents/BundleSDF/assets/{obj_name}_flipped2.obj')

def post_mesh():
    path = "/home/yufeiyang/Documents/BundleSDF/assets_textured/D_shape_video.obj"
    mesh = trimesh.load(path)
    num_faces = len(mesh.faces)
    if num_faces > 10000:
        print("COARSIFIED")
        ratio = 1 - (10000/num_faces)

        root, ext = os.path.splitext(path)   
        new_path = f"{root}_backup{ext}"

        mesh.export(new_path) # make backup of object
        simplified = mesh.simplify_quadric_decimation(ratio)
        simplified.export(path)
        return True
    return

def convert_video():
    
    return
if __name__ == "__main__":
    # view_depth()
    # remove_mesh_offset()
    # cam_in_ob_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/Y_shape/cam_in_ob"
    # view_transformation(cam_in_ob_path)
    # view_basic_frame()

    # joint_config = np.load('/home/yufeiyang/Documents/BundleSDF/arm_data/A_shape/joint_config.npy')
    # joint_config = joint_config[:5, :]
    # print(joint_config)
    # np.save('/home/yufeiyang/Documents/BundleSDF/arm_data/A_shape/joint_config2.npy', joint_config)
    # debug()
    # debug()
    # post_mesh()
    convert_video()