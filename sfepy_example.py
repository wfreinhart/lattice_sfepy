from __future__ import print_function
from __future__ import absolute_import
import numpy as nm

import sys

sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson


# set the total loading (negative = compression, positive = tension)
loading = -2e6

raw_mesh = Mesh.from_file('meshes/strut_test.msh')  # load the gmesh file
data = list(raw_mesh._get_io_data(cell_dim_only=[3]))  # strip non-3d elements
mesh = Mesh.from_data(raw_mesh.name, *data)
domain = FEDomain('domain', mesh)

min_z, max_z = domain.get_mesh_bounding_box()[:, 2]
eps = 1e-4 * (max_z - min_z)
omega = domain.create_region('Omega', 'all')
gamma1 = domain.create_region('Gamma1',
                              'vertices in z < %.10f' % (min_z + eps),
                              'vertex')
gamma2 = domain.create_region('Gamma2',
                              'vertices in z > %.10f' % (max_z - eps),
                              'vertex')

field = Field.from_args('fu', nm.float64, 'vector', omega,
                        approx_order=2)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')

m = Material('m', D=stiffness_from_youngpoisson(dim=3, young=6.8e10, poisson=0.36), rho=2700.0)
load = Material('Load', values={'.val': [[0.0, 0.0, loading/len(gamma2.vertices)] for vert in gamma2.vertices]})

integral = Integral('i', order=3)
integral0 = Integral('i', order=0)

t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
t2 = Term.new('dw_point_load(Load.val, v)', integral0, gamma2, Load=load, v=v)
eq1 = Equation('balance', t1 - t2)
eqs = Equations([eq1])

fix_bot = EssentialBC('fix_bot', gamma1, {'u.all': 0.0})
fix_top = EssentialBC('fix_top', gamma2, {'u.[0,1]': 0.0})

ls = ScipyDirect({})

nls_status = IndexedStruct()
nls = Newton({}, lin_solver=ls, status=nls_status)

pb = Problem('elasticity', equations=eqs)
pb.save_regions_as_groups('regions')

pb.set_bcs(ebcs=Conditions([fix_bot, fix_top]))

pb.set_solver(nls)

status = IndexedStruct()
state = pb.solve(status=status)

print('Nonlinear solver status:\n', nls_status)
print('Stationary solver status:\n', status)

pb.save_state('linear_elasticity.vtk', state)

show = True
if show:
    view = Viewer('linear_elasticity.vtk')
    view(vector_mode='warp_norm', rel_scaling=2,
         is_scalar_bar=True, is_wireframe=True)
