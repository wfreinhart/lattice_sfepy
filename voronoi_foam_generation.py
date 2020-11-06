import gmsh
import sys
import numpy as np
from scipy import spatial
from sfepy.discrete.fem import Mesh
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


def is_inside(p, L, tol=1e-3):
    return np.all(p >= -tol) and np.all(p - L <= tol)


def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4


def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

mesh_name = 'voronoi_foam'
gmsh.model.add(mesh_name)

L = np.array([50, 100, 25])  # design domain

thickness = 1
n_cells = 50
r = (np.random.random([n_cells, 2]) - 0.25) * (2*L[:2])
vor = spatial.Voronoi(r)

# voro_cells = pyvoro.compute_voronoi(r, L[:2], np.max(L[:2]), periodic=[True, True])

bb = np.array([[0, 0],
               [1, 0],
               [1, 1],
               [0, 1],
               [0, 0]]) * L[:2]

z_axis = np.array([0, 0, 1])
txy = np.array([thickness/4, thickness/2])

panel_width = 5
boxes = []
nan_points = np.zeros([0, 2])
nan_points_trunc = np.zeros([0, 2])
for i, j in [x for x in vor.ridge_vertices if -1 not in x]:
    # vor.vertices[j]
    p = vor.vertices[i]
    p_in = is_inside(p, L[:2])
    q = vor.vertices[j]
    q_in = is_inside(q, L[:2])
    nan_points = np.vstack([nan_points, np.vstack([p, q, [np.nan] * 2])])
    # find intersections with box
    l = np.vstack([p, q])
    x = np.vstack(intersection(l[:, 0], l[:, 1], bb[:, 0], bb[:, 1])).T.reshape(-1, 2)
    if x.shape[0] > 0:
        x = np.vstack([row for row in x if is_inside(row, L[:2])])
    print(i, j, x)
    if x.shape[0] == 2:
        verts = np.vstack(x)
    elif x.shape[0] == 1 and (q_in or p_in):
        ins = p if p_in else q
        verts = np.vstack([ins, x])
    elif p_in and q_in:
        verts = np.vstack([p, q])
    else:
        continue
    vec = verts[1] - verts[0]
    length = np.linalg.norm(vec)
    angle = np.arctan2(vec[1], vec[0])
    bxy = verts[0] - txy
    boxes.append(gmsh.model.occ.addBox(*bxy, 0, length+thickness/2, thickness, L[2]))
    gmsh.model.occ.rotate([(3, boxes[-1])], *verts[0], 0, *z_axis, angle)
    nan_points_trunc = np.vstack([nan_points_trunc, np.vstack([verts, [np.nan] * 2])])


make_plot = True
if make_plot:
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(*nan_points.T)
    ax.plot(*nan_points_trunc.T, ':')
    ax.set_xlim(-L[0]*0.5, L[0]*1.5)
    ax.set_ylim(-L[1]*0.5, L[1]*1.5)
    ax.plot(*r.T, '.')
    ax.plot(*bb.T)
    ax.set_aspect('equal')
    ax.figure.show()

# create the side plates to enclose the 2D voronoi foam
t2 = thickness  # * np.sqrt(2)
boxes.append(gmsh.model.occ.addBox(    -t2/2,     -t2/2, 0,
                                   L[0]+t2, t2, L[2]))  # bot
boxes.append(gmsh.model.occ.addBox(    -t2/2, L[1]-t2/2, 0,
                                   L[0]+t2, t2, L[2]))  # top
boxes.append(gmsh.model.occ.addBox(    -t2/2,     -t2/2, 0,
                                   t2, L[1]+t2, L[2]))  # left
boxes.append(gmsh.model.occ.addBox(L[0]-t2/2, -t2/2, 0,
                                   t2, L[1]+t2, L[2]))  # right

aggTags, tagsMap = gmsh.model.occ.fuse([(3, s) for s in boxes[1:]], [(3, boxes[0])])

# rotate so the z axis is "up"
x_axis = [1, 0, 0]
gmsh.model.occ.rotate(aggTags, 0, 0, 0, *x_axis, np.pi/2)

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

lcar1 = 2
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
gmsh.write("meshes/%s.stl" % mesh_name)

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

raw_mesh = Mesh.from_file("meshes/%s.msh" % mesh_name)  # load the gmesh file
data = list(raw_mesh._get_io_data(cell_dim_only=[3]))  # strip non-3d elements
mesh = Mesh.from_data(raw_mesh.name, *data)
mesh.write("meshes/%s.vtk" % mesh_name)
