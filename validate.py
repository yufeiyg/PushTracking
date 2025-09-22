import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import os
import time
import glob
import datetime

code_dir = os.path.dirname(os.path.realpath(__file__))
print(code_dir)

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


# === 1. Load the OBJ mesh ===
mesh_path = f'/home/yufeiyang/Documents/BundleSDF/assets_textured/baby_toy.obj'
# mesh_path = "auto_rotate_mesh.obj"
scene_or_mesh = trimesh.load(mesh_path)
world_T_cam = np.array([[ 0.9929435 ,  0.00945858, -0.11821058,  0.42198977],
                        [ 0.05946506,  0.82272298,  0.56532363, -0.30166237],
                        [ 0.10260172, -0.56836382,  0.81635498, -0.17893198],
                        [ 0.,          0.,          0.,          1.]])
if isinstance(scene_or_mesh, trimesh.Scene):
    trimesh_mesh = scene_or_mesh.dump(concatenate=True)  # merge into one Trimesh
else:
    trimesh_mesh = scene_or_mesh
# === 2. Create a MeshCat visualizer ===
vis = meshcat.Visualizer().open()
vis.delete()  # Clear the scene

world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')  

# Create a MeshCat mesh object from Trimesh geometry
vertices = trimesh_mesh.vertices.astype(np.float32)
faces = trimesh_mesh.faces.astype(np.uint32)
meshcat_mesh = g.TriangularMeshGeometry(vertices, faces)

# Set the object in MeshCat
vis["object"].set_object(meshcat_mesh, g.MeshLambertMaterial(color=0x00FF00))
# vis["object"].set_transform(world_T_object)

# === 3. Load ob_in_cam poses ===
pose_folder = "/home/yufeiyang/Documents/BundleSDF/foundationPose/plus_video/ob_in_cam"
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
T0 = poses[0]
T0_inv = np.linalg.inv(T0)
adjusted_poses = [T for T in poses][5459:5463]

time.sleep(1) 
# === 5. Animate the mesh using rebased poses ===
for i, cam_T_object in enumerate(adjusted_poses):
    # cam_T_object = np.eye(4)  # Reset to identity for each frame
    world_T_object = world_T_cam @ cam_T_object
    # print(world_T_object)
    # vis["object"].set_transform(cam_T_object) # step 1
    vis["object"].set_transform(world_T_object)
    print(f"Showing frame {i+5459} from file")
    time.sleep(1)  # Adjust playback speed here