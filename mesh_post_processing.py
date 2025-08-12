import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import os
import time
import glob
import re
import pymeshfix
import pymeshlab
import open3d as o3d



# def numerical_sort(value):
#     # Extract the first number found in the filename
#     match = re.search(r'(\d+)', os.path.basename(value))
#     return int(match.group(1)) if match else -1

# def load_matrix_from_txt(path):
#     data = np.loadtxt(path)
#     if data.size != 16:
#         raise ValueError(f"File {path} does not contain a 4x4 matrix.")
#     return data.reshape(4, 4)

# code_dir = os.path.dirname(os.path.realpath(__file__))

# # === 1. Load the OBJ mesh and pose ===
# mesh_path = f'{code_dir}/debug_output/textured_mesh.obj'
# pose_folder = f"{code_dir}/debug_output/ob_in_cam"
# pose_files = sorted(glob.glob(os.path.join(pose_folder, "*.txt")), key=numerical_sort)


# # Load all poses first
# poses = [load_matrix_from_txt(pf) for pf in pose_files]
# last_frame_pose = poses[-1]

# scene_or_mesh = trimesh.load(mesh_path)
# # Load the camera extrinsics
# world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
#                         [-0.99248708, 0.11664051, -0.03693756, 0.],
#                         [-0.06717635, -0.77182713, -0.63227385, 0.35],
#                         [0., 0., 0., 1.]])
# trimesh_mesh = scene_or_mesh
# world_T_object = world_T_cam @ last_frame_pose
# print(world_T_object)
# # === 2. Create a MeshCat visualizer ===
# vis = meshcat.Visualizer().open()
# vis.delete()  # Clear the scene


# trimesh_mesh.apply_transform((world_T_object))

# trimesh_mesh.apply_translation(-trimesh_mesh.centroid)

# # Create a MeshCat mesh object from Trimesh geometry
# vertices = trimesh_mesh.vertices.astype(np.float32)
# faces = trimesh_mesh.faces.astype(np.uint32)
# meshcat_mesh = g.TriangularMeshGeometry(vertices, faces)

# # test watertight
# if not trimesh_mesh.is_watertight:
#     print("Raw mesh from BundleSDF is not watertight")
#     trimesh_mesh.merge_vertices()
#     trimesh_mesh.remove_duplicate_faces()
#     trimesh_mesh.remove_degenerate_faces()
#     trimesh_mesh.remove_unreferenced_vertices()
#     meshfix = pymeshfix.MeshFix(trimesh_mesh.vertices, trimesh_mesh.faces)
#     meshfix.repair()

#     fixed_mesh = trimesh.Trimesh(meshfix.v, meshfix.f)
# print("Is the mesh watertight?", fixed_mesh.is_watertight)
# # Set the object in MeshCat
# vis["object"].set_object(meshcat_mesh, g.MeshLambertMaterial(color=0x00FF00))
# vis["object"].set_transform(np.eye(4))
# breakpoint()
# # export the mesh
# fixed_mesh.export('auto_rotate_mesh.obj')

def smooth_mesh(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)

    # Laplacian smoothing
    ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=3)

    # Optional: HC smoothing (preserves features better)
    ms.apply_filter("apply_coord_hc_laplacian_smoothing")
    ms.save_current_mesh("smoothed_mesh.obj")

def flat_top(path):
    tol = 1e-5
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)

    # Find Z min and max
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()

    # Snap bottom vertices
    bottom_mask = np.isclose(vertices[:, 2], z_min, atol=tol)
    vertices[bottom_mask, 2] = z_min

    # Snap top vertices
    top_mask = np.isclose(vertices[:, 2], z_max, atol=tol)
    vertices[top_mask, 2] = z_max

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh("mesh_flat.obj", mesh)
    return mesh


def sharpen_edges(path):

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()

    # Save a copy for reference
    original_vertices = np.asarray(mesh.vertices).copy()

    # Smooth slightly
    mesh = mesh.filter_smooth_taubin(number_of_iterations=20)
    mesh.compute_vertex_normals()

    # Unsharp mask: amplify difference from smoothed version
    sharpen_amount = 0.5
    mesh_vertices = np.asarray(mesh.vertices)
    original_vertices += (original_vertices - mesh_vertices) * sharpen_amount

    mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
    o3d.io.write_triangle_mesh("mesh_sharpened.obj", mesh)
    return


if __name__ == "__main__":
    mesh_name = "A_shape.obj"
    flat_top(mesh_name)
    smooth_mesh("mesh_flat.obj")
    sharpen_edges("smoothed_mesh.obj")
    # fillup("smoothed_mesh.obj")
    # flat_top("smoothed_mesh.obj")
    # smooth_mesh(mesh_name)