import trimesh
import numpy as np

# Load mesh
mesh = trimesh.load('/home/yufeiyang/Documents/BundleSDF/debug_output/textured_mesh.obj')
angle_thresh_degrees = 5
epsilon = 1e-3
thickness = 1.0

# Step 1: Find largest coplanar group
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

largest_group = max(groups, key=lambda g: mesh.area_faces[g].sum())
flat_faces = mesh.faces[largest_group]

# Step 2: Extract boundary from the flat surface
flat_mesh = mesh.submesh([largest_group], append=True)
flat_mesh.merge_vertices()  # cleanup
boundary_edges = flat_mesh.facets_boundary

if len(boundary_edges) == 0:
    raise ValueError("No boundary found; surface may already be closed.")

# Step 3: Form a polygon from boundary loop
boundary_loop = trimesh.path.polygons.edges_to_polygons(
    edges=boundary_edges, 
    vertices=flat_mesh.vertices
)[0]

# Step 4: Extrude polygon downward
extruded = trimesh.creation.extrude_polygon(boundary_loop, height=thickness)

# Step 5: Align extrusion with surface
normal = flat_mesh.face_normals.mean(axis=0)
normal /= np.linalg.norm(normal)
translation = flat_mesh.centroid + normal * (-thickness)
T = trimesh.geometry.align_vectors([0, 0, 1], normal)
T[:3, 3] = translation
extruded.apply_transform(T)

# Step 6: Combine original mesh and extruded solid
combined = trimesh.util.concatenate([mesh, extruded])
combined.merge_vertices()

# Export or display
combined.export('watertight_output.obj')
# combined.show()
