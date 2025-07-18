import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import os
import time
import glob

code_dir = os.path.dirname(os.path.realpath(__file__))
print(code_dir)

# === 1. Load the OBJ mesh ===
mesh_path = f'/home/yufeiyang/Documents/BundleSDF/debug_output/textured_mesh.obj'
# mesh_path = "auto_rotate_mesh.obj"
scene_or_mesh = trimesh.load(mesh_path)
# TODO change this

world_T_cam = np.array([[-0.10225815, -0.6250423, 0.77386394, -0.27],
                        [-0.99248708, 0.11664051, -0.03693756, 0.],
                        [-0.06717635, -0.77182713, -0.63227385, 0.35],
                        [0., 0., 0., 1.]])
trimesh_mesh = scene_or_mesh

# === 2. Create a MeshCat visualizer ===
vis = meshcat.Visualizer().open()
vis.delete()  # Clear the scene


world_T_object = np.array([[-0.65788279, -0.69395063, -0.2926138,   0.14579146],
                       [-0.68949708,  0.39868771,  0.60468328,  0.10442389],
                       [-0.30295889,  0.59956706, -0.7407667,   0.01661531],
                       [ 0.,          0.,          0.,          1.        ]])
trimesh_mesh.apply_transform(world_T_object)
trimesh_mesh.apply_translation(-trimesh_mesh.centroid)

# Create a MeshCat mesh object from Trimesh geometry
vertices = trimesh_mesh.vertices.astype(np.float32)
faces = trimesh_mesh.faces.astype(np.uint32)
meshcat_mesh = g.TriangularMeshGeometry(vertices, faces)

# Set the object in MeshCat
vis["object"].set_object(meshcat_mesh, g.MeshLambertMaterial(color=0x00FF00))
vis["object"].set_transform(np.eye(4))
# vis["object"].set_transform(world_T_object)

breakpoint()
# === 3. Load ob_in_cam poses ===
pose_folder = "/home/yufeiyang/Documents/BundleSDF/debug_output/ob_in_cam"
import re
def numerical_sort(value):
    # Extract the first number found in the filename
    match = re.search(r'(\d+)', os.path.basename(value))
    return int(match.group(1)) if match else -1

pose_files = sorted(glob.glob(os.path.join(pose_folder, "*.txt")), key=numerical_sort)

def load_matrix_from_txt(path):
    data = np.loadtxt(path)
    if data.size != 16:
        raise ValueError(f"File {path} does not contain a 4x4 matrix.")
    return data.reshape(4, 4)

# Load all poses first
poses = [load_matrix_from_txt(pf) for pf in pose_files]

# # === 4. Rebase poses relative to first pose ===
# T0 = poses[0]
# T0_inv = np.linalg.inv(T0)
adjusted_poses = [T for T in poses][-2:]


# === 5. Animate the mesh using rebased poses ===
for i, cam_T_object in enumerate(adjusted_poses):
    # cam_T_object = np.eye(4)  # Reset to identity for each frame
    world_T_object = world_T_cam @ cam_T_object
    print(world_T_object)
    # vis["object"].set_transform(cam_T_object) # step 1
    vis["object"].set_transform(world_T_object)
    print(f"Showing frame {i} from file: {pose_files[i]}")
    time.sleep(0.1)  # Adjust playback speed here