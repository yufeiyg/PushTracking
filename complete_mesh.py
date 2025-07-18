import trimesh
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from trimesh.transformations import rotation_matrix
from trimesh.smoothing import filter_laplacian
from trimesh.smoothing import filter_taubin
import pymeshfix



# def duplicate_top_to_bottom(mesh, thickness=1.0, normal_threshold=0.9):
    # # Step 1: Find the top surface (assuming 'top' is along the mesh's dominant normal)
    # # Compute face normals and centroid
    # face_normals = mesh.face_normals
    # face_centroids = mesh.triangles.mean(axis=1)
    # # breakpoint()
    
    # # Determine up direction (using PCA for better robustness)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=1).fit(mesh.vertices)
    # up_direction = pca.components_[0]
    
    # # Find top-facing faces
    # dot_products = np.dot(face_normals, up_direction)
    # top_face_mask = dot_products > normal_threshold
    # top_faces = mesh.faces[top_face_mask]
    
    # # Get all vertices from top faces
    # top_vertex_indices = np.unique(top_faces.flatten())
    # top_vertices = mesh.vertices[top_vertex_indices]
    
    # # Find height range of top surface
    # heights = np.dot(top_vertices, up_direction)
    # min_height = np.min(heights)
    # max_height = np.max(heights)
    
    # # Find ALL vertices within thickness range (not just surface vertices)
    # all_heights = np.dot(mesh.vertices, up_direction)
    # thickness_mask = (all_heights >= (max_height - thickness)) & (all_heights <= max_height)
    # thick_vertex_indices = np.where(thickness_mask)[0]
    
    # # Create vertex mapping for new mesh
    # vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(thick_vertex_indices)}
    
    # # Get all faces that use ONLY these vertices
    # thick_faces = []
    # for face in mesh.faces:
    #     if all(v in thick_vertex_indices for v in face):
    #         thick_faces.append([vertex_map[v] for v in face])
    
    # if not thick_faces:
    #     raise ValueError("No faces found in thickness range - try increasing thickness")
    
    # # Create new mesh
    # thick_surface = trimesh.Trimesh(
    #     vertices=mesh.vertices[thick_vertex_indices],
    #     faces=np.array(thick_faces)
    # )
    
    # return thick_surface

def duplicate_top_to_bottom(mesh, thickness=0.05, angle_thresh_degrees=10, epsilon=0.005):
    # 1. Group faces by normal similarity
    face_normals = mesh.face_normals
    angle_thresh_cos = np.cos(np.radians(angle_thresh_degrees))
    groups = []
    used = np.zeros(len(face_normals), dtype=bool)

    for i, n_i in enumerate(face_normals):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(face_normals)):
            if used[j]:
                continue
            n_j = face_normals[j]
            if np.dot(n_i, n_j) > angle_thresh_cos:
                group.append(j)
                used[j] = True
        groups.append(group)

    # 2. Pick largest group by total face area
    largest_group = max(groups, key=lambda g: mesh.area_faces[g].sum())
    flat_faces = mesh.faces[largest_group]
    flat_face_indices = np.unique(flat_faces)

    # 3. Filter out outlier vertices not lying on the main surface plane
    flat_vertices = mesh.vertices[flat_face_indices]
    normal = face_normals[largest_group].mean(axis=0)
    normal /= np.linalg.norm(normal)
    centroid = flat_vertices.mean(axis=0)
    distances = np.dot(flat_vertices - centroid, normal)
    inlier_mask = np.abs(distances) < epsilon
    flat_face_indices = flat_face_indices[inlier_mask]
    flat_vertices = mesh.vertices[flat_face_indices]

    # 4. Create bottom layer
    bottom_vertices = flat_vertices - thickness * normal
    bottom_start_index = len(mesh.vertices)
    new_vertices = np.vstack([mesh.vertices, bottom_vertices])

    # 5. Create bottom faces (reversed winding)
    bottom_faces = []
    for face in flat_faces:
        try:
            indices = [np.where(flat_face_indices == v)[0][0] for v in face]
            bottom_face = [bottom_start_index + idx for idx in indices]
            bottom_faces.append(bottom_face[::-1])
        except IndexError:
            continue  # skip if any vertex was filtered out

    # 6. Build side faces (connect top and bottom edges)
    edge_set = set()
    for face in flat_faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edge_set.add(edge)

    side_faces = []
    for a, b in edge_set:
        try:
            ia = np.where(flat_face_indices == a)[0][0]
            ib = np.where(flat_face_indices == b)[0][0]
        except IndexError:
            continue  # skip if vertex was filtered out
        a_bot = bottom_start_index + ia
        b_bot = bottom_start_index + ib
        side_faces.append([a, b, b_bot])
        side_faces.append([a, b_bot, a_bot])
        

    # 7. Combine all faces
    new_faces = np.vstack([mesh.faces, bottom_faces, side_faces])

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)


# mve mesh to the origin
# closed_mesh = trimesh.load_mesh("/home/yufeiyang/Documents/BundleSDF/debug_output/textured_mesh.obj")
closed_mesh = trimesh.load_mesh("/home/yufeiyang/Documents/BundleSDF/debug_output/textured_mesh.obj")

# closed_mesh = duplicate_top_to_bottom(closed_mesh)



T = closed_mesh.bounding_box_oriented.primitive.transform
T_inv = np.linalg.inv(T)
closed_mesh.apply_transform(T_inv)
closed_mesh.apply_translation(-closed_mesh.centroid)

# rotate mesh -90 degrees around the z-axis
angle_rad = np.deg2rad(-90)
rotation = rotation_matrix(
    angle_rad,              # angle in radians
    direction=[0, 1, 0],    # Y-axis
    point=[0, 0, 0]         # rotate around origin
)
closed_mesh.apply_transform(rotation)

# # rotate the mesh 180 degrees around the z axis
# angle_rad = np.deg2rad(180)
# rotation = rotation_matrix(
#     angle_rad,              # angle in radians
#     direction=[1, 0, 0],    # Z-axis
#     point=[0, 0, 0]         # rotate around origin
# )
# closed_mesh.apply_transform(rotation)

# closed_mesh.fill_holes()

# closed_mesh.merge_vertices()
# closed_mesh.remove_duplicate_faces()
# closed_mesh.remove_degenerate_faces()
# closed_mesh.remove_unreferenced_vertices()
# meshfix = pymeshfix.MeshFix(closed_mesh.vertices, closed_mesh.faces)
# meshfix.repair()

# closed_mesh = trimesh.Trimesh(meshfix.v, meshfix.f)



print("Is the mesh watertight?", closed_mesh.is_watertight)
closed_mesh.export("your_mesh_centered.obj")