import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
def view_depth():
    depth_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/T_shape_single/ob_0000001/depth_enhanced/00001.png"
    # depth_path = "/home/yufeiyang/Documents/FoundationPose/ycbv/ref_views_4/ob_0000001/depth_enhanced/0000000.png"
    # visualize this depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32)  # Convert to meters
    breakpoint()
    cv2.imshow("Depth Image", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_mesh_offset():
    mesh = trimesh.load('/home/yufeiyang/Documents/BundleSDF/assets/plus.obj')
    mesh.apply_translation(-mesh.centroid)
    mesh.export(f'/home/yufeiyang/Documents/BundleSDF/assets/plus1.obj')

def draw_transformation(trans, ax):
    # Given cam_in_ob, a 4 by 4 transformation matrix, draw this as a triad with matplotlib
    # also draw the origin

    # Draw the camera coordinate system
    origin = trans[:3, 3]
    x_axis = trans[:3, 0]
    y_axis = trans[:3, 1]
    z_axis = trans[:3, 2]

    ax.quiver(*origin, *x_axis, color='r', length=0.3)
    ax.quiver(*origin, *y_axis, color='g', length=0.3)
    ax.quiver(*origin, *z_axis, color='b', length=0.3)

    ax.set_xlim([-0.5, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    return

def view_transformation(cam_in_ob):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_transformation(cam_in_ob, ax)
    draw_transformation(np.eye(4), ax)
    plt.show()

if __name__ == "__main__":
    # view_depth()
    # remove_mesh_offset()
    matrix_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/T_shape/ob_0000001/cam_in_ob/00001.txt"
    cam_in_ob = np.loadtxt(matrix_path)
    view_transformation(cam_in_ob)