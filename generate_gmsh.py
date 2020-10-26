import gmsh
import sys
import numpy as np
from sfepy.discrete.fem import Mesh
# from scipy.spatial.transform import Rotation

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

mesh_name = 'strut_test'
gmsh.model.add(mesh_name)

# octet truss cell (one eighth, to be expanded by symmetry)
points = np.array([[0, 0, 0],
                   [0.5, 0, 0.5],
                   [0, 0.5, 0.5],
                   [0.5, 0.5, 0]])
points -= 0.5 * np.ones(3)
edges = np.array([[0, 1],
                  [0, 2],
                  [0, 3],
                  [1, 2],
                  [1, 3],
                  [2, 3]])

# create the cubic symmetry
for dim in range(3):
    n = np.zeros([3, 1])
    n[dim] = 1
    M = np.eye(3) - 2 * n @ n.T  # I
    # angles = np.zeros(3)
    # angles[dim] = np.pi/2
    # R = Rotation.from_euler('xyz', angles).as_matrix()
    edges = np.vstack([edges, edges + len(points)])
    points = np.vstack([points, points @ M])

points += 0.5 * np.ones(3)

# remove duplicates and renumber nodes/edges
unq, inv = np.unique(points, axis=0, return_inverse=True)
umap = {i: x for i, x in enumerate(inv)}
points = unq
edges = np.array([[umap[i], umap[j]] for i, j in edges])

# create a cylinder
# def addCylinder(x, y, z, dx, dy, dz, r, tag=-1, angle=2 * pi):
#     """
#     gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r, tag=-1, angle=2*pi)
#
#     Add a cylinder, defined by the center (`x', `y', `z') of its first circular
#     face, the 3 components (`dx', `dy', `dz') of the vector defining its axis
#     and its radius `r'. The optional `angle' argument defines the angular
#     opening (from 0 to 2*Pi). If `tag' is positive, set the tag explicitly;
#     otherwise a new tag is selected automatically. Return the tag of the
#     cylinder.
#
#     Return an integer value.
#     """
strut_rad = 0.02
struts = []
for i, j in edges:
    struts.append(gmsh.model.occ.addCylinder(*points[i], *(points[j] - points[i]), strut_rad))

# fuse the two cylinders
# def fuse(objectDimTags, toolDimTags, tag=-1, removeObject=True, removeTool=True):
#     """
#     gmsh.model.occ.fuse(objectDimTags, toolDimTags, tag=-1, removeObject=True, removeTool=True)
#
#     Compute the boolean union (the fusion) of the entities `objectDimTags' and
#     `toolDimTags'. Return the resulting entities in `outDimTags'. If `tag' is
#     positive, try to set the tag explicitly (only valid if the boolean
#     operation results in a single entity). Remove the object if `removeObject'
#     is set. Remove the tool if `removeTool' is set.
#
#     Return `outDimTags', `outDimTagsMap'.
#     """
# aggTags = [(3, struts[0])]
# for s in struts[1:]:
#     aggTags, tagsMap = gmsh.model.occ.fuse(aggTags, [(3, s)])

aggTags, tagsMap = gmsh.model.occ.fuse([(3, s) for s in struts[1:]], [(3, struts[0])])

# create the mirrors for cubic symmetry
# def mirror(dimTags, a, b, c, d):
#     """
#     gmsh.model.geo.mirror(dimTags, a, b, c, d)
#
#     Mirror the model entities `dimTag', with respect to the plane of equation
#     `a' * x + `b' * y + `c' * z + `d' = 0.
#     """
# for dim in range(3):
#     abc = np.zeros(3)
#     abc[dim] = 1
#     copyTags = gmsh.model.occ.copy(aggTags)
#     gmsh.model.occ.mirror(copyTags, *abc, -0.5)
#     aggTags, tagsMap = gmsh.model.occ.fuse(aggTags, copyTags)

# create the top and bottom plates for the sandwich panel
overhang = 0.25
plates = []
for i, z in enumerate([np.min(points[:, 2]), np.max(points[:, 2])]):
    delta_z = strut_rad
    if i == 0:
        delta_z *= -1
    plates.append(gmsh.model.occ.addBox(-overhang, -overhang, z, *np.array([1+2*overhang, 1+2*overhang, delta_z])))
gmsh.model.occ.fuse(aggTags, [(3, p) for p in plates])

# Boolean operations with OpenCASCADE always create new entities. By default the
# extra arguments `removeObject' and `removeTool' in `cut()' are set to `True',
# which will delete the original entities.

gmsh.model.occ.synchronize()

# When the boolean operation leads to simple modifications of entities, and if
# one deletes the original entities, Gmsh tries to assign the same tag to the
# new entities. (This behavior is governed by the
# `Geometry.OCCBooleanPreserveNumbering' option.)

# Here the `Physical Volume' definitions can thus be made for the 5 spheres
# directly, as the five spheres (volumes 4, 5, 6, 7 and 8), which will be
# deleted by the fragment operations, will be recreated identically (albeit with
# new surfaces) with the same tags:
gmsh.model.addPhysicalGroup(3, aggTags)

# The tag of the cube will change though, so we need to access it
# programmatically:
# gmsh.model.addPhysicalGroup(3, [ov[-1][1]], 10)

# Creating entities using constructive solid geometry is very powerful, but can
# lead to practical issues for e.g. setting mesh sizes at points, or identifying
# boundaries.

# To identify points or other bounding entities you can take advantage of the
# `getEntities()', `getBoundary()' and `getEntitiesInBoundingBox()' functions:

lcar1 = .03
# lcar2 = .0005
# lcar3 = .055

# Assign a mesh size to all the points:
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar1)

# Override this constraint on the points of the five spheres:
# gmsh.model.mesh.setSize(gmsh.model.getBoundary(holes, False, False, True),
#                         lcar3)

# Select the corner point by searching for it geometrically:
# eps = 1e-3
# ov = gmsh.model.getEntitiesInBoundingBox(0.5 - eps, 0.5 - eps, 0.5 - eps,
#                                          0.5 + eps, 0.5 + eps, 0.5 + eps, 0)
# gmsh.model.mesh.setSize(ov, lcar2)

gmsh.model.mesh.generate(3)

gmsh.write("meshes/%s.msh" % mesh_name)
gmsh.write("meshes/%s.vtk" % mesh_name)

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

raw_mesh = Mesh.from_file("meshes/%s.msh" % mesh_name)  # load the gmesh file
data = list(raw_mesh._get_io_data(cell_dim_only=[3]))  # strip non-3d elements
mesh = Mesh.from_data(raw_mesh.name, *data)
mesh.write("meshes/%s.vtk" % mesh_name)
