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
    steps = 1
    # Laplacian smoothing
    ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=1)

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

def rotate_fill_mesh(mesh_file):
  scene_or_mesh = trimesh.load(mesh_file)
  # Load the camera extrinsics
  trimesh_mesh = scene_or_mesh
  # === 2. Create a MeshCat visualizer ===
  trimesh_mesh.apply_translation(-trimesh_mesh.centroid)
#   remove all floating, unconnected fragments
  components = trimesh_mesh.split(only_watertight=False)
  largest_area = max(c.area for c in components)
  threshold = 0.002  # 5% of largest
  big_components = [c for c in components if c.area > largest_area * threshold]
  mesh_clean = trimesh.util.concatenate(big_components)
  mesh_clean.merge_vertices(digits_vertex=2)
  mesh_clean.fill_holes()
  mesh_o3d = o3d.geometry.TriangleMesh(
      o3d.utility.Vector3dVector(mesh_clean.vertices),
      o3d.utility.Vector3iVector(mesh_clean.faces)
  )

    # Weld vertices that fall into the same voxel (size = tolerance)
  tolerance = 0.005  # adjust based on your scale
  mesh_welded = mesh_o3d.simplify_vertex_clustering(voxel_size=tolerance)

#   fill up holes
#   pcd = mesh_welded.sample_points_poisson_disk(5000)
#   pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
#     )
#   pcd.orient_normals_consistent_tangent_plane(30)  # Optional, helps avoid inverted regions
#   mesh_rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
#   mesh_rec = mesh_rec.crop(mesh_welded.get_axis_aligned_bounding_box())

  mesh_welded.compute_vertex_normals()

  # Voxelize (convert to occupancy grid)
  voxel_size = 0.003
  voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_welded, voxel_size=voxel_size)

  # Convert voxel grid to numpy occupancy grid
  voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
  min_bound = voxel_grid.get_min_bound()
  max_bound = voxel_grid.get_max_bound()

    # Create dense 3D volume
  dims = np.max(voxels, axis=0) + 1
  volume = np.zeros(dims, dtype=np.uint8)
  for v in voxels:
      volume[tuple(v)] = 1

    # Run Marching Cubes to extract surface
  from skimage import measure
  verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5, spacing=[voxel_size]*3)

  mesh_filled = o3d.geometry.TriangleMesh()
  mesh_filled.vertices = o3d.utility.Vector3dVector(verts + min_bound)  # shift back
  mesh_filled.triangles = o3d.utility.Vector3iVector(faces)
  mesh_filled.compute_vertex_normals()

  o3d.io.write_triangle_mesh("fixed_mesh.obj", mesh_filled)


if __name__ == "__main__":
    name = "C_shape"
    mesh_name = f"/home/yufeiyang/Documents/BundleSDF/arm_data/{name}/model/model.obj"
    rotate_fill_mesh(mesh_file=mesh_name)
    smooth_mesh("fixed_mesh.obj")
    # --- Step 1: Load mesh with trimesh
    mesh = trimesh.load("smoothed_mesh.obj")
    print("Vertices:", len(mesh.vertices), " Faces:", len(mesh.faces))

    # --- Step 2: Convert to Open3D for vertex clustering (welding)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_o3d.compute_vertex_normals()

    # Weld vertices within tolerance
    mesh_o3d = mesh_o3d.simplify_vertex_clustering(voxel_size=0.01)

    # Convert back to Trimesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=False
    )

    # --- Step 3: Clean mesh
    # mesh.remove_duplicate_faces()
    # mesh.remove_degenerate_faces()
    # mesh.remove_unreferenced_vertices()

    # --- Step 4: Remove floating junk
    components = mesh.split(only_watertight=False)
    largest_area = max(c.area for c in components)
    keep = [c for c in components if c.area > 0.001 * largest_area]
    mesh = trimesh.util.concatenate(keep)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    pcd = mesh_o3d.sample_points_poisson_disk(50000)
    pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    pcd.orient_normals_consistent_tangent_plane(30)  # Optional, helps avoid inverted regions
    mesh_rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    mesh_rec = mesh_rec.crop(mesh_o3d.get_axis_aligned_bounding_box())

    # print("After cleaning - Vertices:", len(mesh_filled.vertices), " Faces:", len(mesh_filled.faces))

    # --- Step 5: Run MeshFix to enforce watertightness
    mesh_connected = trimesh.Trimesh(
        vertices=np.asarray(mesh_rec.vertices),
        faces=np.asarray(mesh_rec.triangles),
        process=False
    )
    meshfix = pymeshfix.MeshFix(mesh_connected.vertices, mesh_connected.faces)
    meshfix.repair(verbose=True, joincomp=True, remove_smallest_components=False)
    fixed_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
    
    # --- Step 6: Save result
    print("Is the final mesh watertight?", fixed_mesh.is_watertight)
    fixed_mesh.export(f"{name}_new.obj")
