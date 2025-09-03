import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
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

def view_basic_frame():
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
    draw_transformation(BASE_T_W, ax, "base_T_world")
    draw_transformation(OBJ_T_W, ax, "obj_T_world")
    # draw_transformation(cam_T_W, ax, "cam_T_W")
    # draw_transformation(np.linalg.inv(ee_T_cam), ax, "ee_T_cam")
    # draw_transformation(ee_T_W, ax, "ee_T_W")
    # draw_transformation(ee_T_Base, ax, "ee_T_Base")
    plt.show()

if __name__ == "__main__":
    # view_depth()
    remove_mesh_offset()
    # cam_in_ob_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/Y_shape/cam_in_ob"
    # view_transformation(cam_in_ob_path)
    # view_basic_frame()

    # joint_config = np.load('/home/yufeiyang/Documents/BundleSDF/arm_data/A_shape/joint_config.npy')
    # joint_config = joint_config[:5, :]
    # print(joint_config)
    # np.save('/home/yufeiyang/Documents/BundleSDF/arm_data/A_shape/joint_config2.npy', joint_config)