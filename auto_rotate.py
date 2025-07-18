import trimesh
import pymeshfix
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


AXIS_SCALING = 0.1
CAMERA_MARKER_COLORS = {'cam0': '#ff0000', 'world': '#00ff00', 'cam2': '#0000ff',
                        'realsense': '#ffff00'}
CAMERA_AXES_IN_CAMERA_FRAME = AXIS_SCALING * np.array([[[0, 0, 0], [1, 0, 0]],
                                                       [[0, 0, 0], [0, 1, 0]],
                                                       [[0, 0, 0], [0, 0, 1]]])

def transform_point_coordinates_given_pose(
        points_in_A: np.ndarray, pose_A_in_B: np.ndarray) -> np.ndarray:
    """Transform a set of points represented in frame A to being represented in
    frame B, given the pose of frame A in frame B.  Interpret the pose as having
    order [x, y, z, qx, qy, qz, qw]."""
    assert points_in_A.ndim == 2 and points_in_A.shape[1] == 3
    assert pose_A_in_B.ndim == 1 and pose_A_in_B.shape[0] == 7

    xyz = pose_A_in_B[:3]
    quat_xyzw = pose_A_in_B[3:]

    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

    points_in_B = (rotation_matrix @ points_in_A.T).T + xyz
    return points_in_B

def axis_angle_to_quat(axis_angle):
    """Convert axis-angle to quaternion.  Returns xyzw ordering."""
    return R.from_rotvec(axis_angle).as_quat()

def inspect_camera_poses_and_images():
    """Plot the camera locations in 3D."""
    translation = np.array([-0.27, 0, 0.35])
    axis_angles = np.array([np.deg2rad(-90), np.deg2rad(103), np.deg2rad(-45)]) # TODO change this 

    world = np.array([0, 0, 0]).reshape(1, 3)
    cam0 = translation.reshape(1, 3)

    world_to_cam0 = np.concatenate((world, cam0), axis=0)

    def plot_camera_triad(cam_trans, cam_axis_angle, label):
        pose = np.hstack((cam_trans,
                          axis_angle_to_quat(cam_axis_angle)))
        quat = pose[3:]
        rot = R.from_quat(quat).as_matrix()
        transform_mat = np.hstack((rot, cam_trans.reshape(3, 1)))
        transform_mat = np.vstack((transform_mat, [0, 0, 0, 1]))
        cam_axes = transform_point_coordinates_given_pose(
            CAMERA_AXES_IN_CAMERA_FRAME.reshape(6,3), pose).reshape(3,2,3)
        plt.plot(cam_axes[0, :, 0], cam_axes[0, :, 1], cam_axes[0, :, 2],
                 color='#ff0000', linewidth=5)
        plt.plot(cam_axes[1, :, 0], cam_axes[1, :, 1], cam_axes[1, :, 2],
                 color='#00ff00', linewidth=5)
        plt.plot(cam_axes[2, :, 0], cam_axes[2, :, 1], cam_axes[2, :, 2],
                 color='#0000ff', linewidth=5)
        plt.plot(cam_trans[0], cam_trans[1], cam_trans[2], marker='o',
                 color=CAMERA_MARKER_COLORS[label], markersize=10, label=label)
        return transform_mat

    # Be prepared to return all the figures.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(world_to_cam0[:, 0], world_to_cam0[:, 1], world_to_cam0[:, 2])
    cam_in_world = plot_camera_triad(translation, axis_angles, 'cam0')
    origin_translation = np.array([0, 0, 0])
    origin_axis_angles = np.array([0., 0., 0.])
    plot_camera_triad(origin_translation, origin_axis_angles, 'world')


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()

    ax.set_box_aspect([np.ptp(arr) for arr in \
                      [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    return cam_in_world

if __name__ == "__main__":
    cam_in_world = inspect_camera_poses_and_images()
    print(cam_in_world)
    plt.show()
    # last frame cam_in_ob

    # ob_in_cam = np.array([[0.7719424963, -0.3650070131, -0.5204561949, -0.1237618625],
    #                       [0.5646131635,  0.01748950779,  0.8251703978, 0.009608156979],
    #                      [-0.2920904458, -0.9308404922,  0.2195886225, 0.5286992788],
    #                       [0, 0, 0, 1]])
    ob_in_cam = np.array([[1, 0, 0, -0.03601440042],
                          [0, 1, 0, -0.05679904297],
                          [0, 0, 1, 0.4646891356],
                          [0, 0, 0, 1]])
    closed_mesh = trimesh.load_mesh("/home/yufeiyang/Documents/BundleSDF/debug_output/textured_mesh.obj")

    # T = ob_in_cam # for step 4
    T = cam_in_world @ ob_in_cam # for step 4
    closed_mesh.apply_transform(np.linalg.inv(T))

    # offset = closed_mesh.bounding_box_oriented.primitive.transform
    # closed_mesh.apply_translation(-closed_mesh.centroid)

    closed_mesh.export('auto_rotate_mesh.obj')

