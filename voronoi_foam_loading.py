from __future__ import print_function
from __future__ import absolute_import
import numpy as np

import sys

sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mechanics.tensors import get_von_mises_stress

mesh = Mesh.from_file('meshes/voronoi_foam.vtk')
domain = FEDomain('domain', mesh)

min_z, max_z = domain.get_mesh_bounding_box()[:, 2]
eps = 5
omega = domain.create_region('Omega', 'all')
bot = domain.create_region('Bot',
                           'vertices in z < %.10f' % (min_z + eps),
                           'vertex')
top = domain.create_region('Top',
                           'vertices in z > %.10f' % (max_z - eps),
                           'vertex')

field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=1)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')

# these are for stainless steel 316L
m = Material('m', D=stiffness_from_youngpoisson(dim=3, young=1.93e9, poisson=0.275), rho=8000.0)

integral = Integral('i', order=1)

t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
eq1 = Equation('balance_of_forces', t1)
eqs = Equations([eq1])


# materials = {
#     'solid': ({'K': 1e3, # bulk modulus
#                'mu': 20e0, # shear modulus of neoHookean term
#                'kappa': 10e0, # shear modulus of Mooney-Rivlin term
#                },),
# }
# equations = {
#     'balance': """dw_ul_he_neohook.3.Omega( solid.mu, v, u )
#                 + dw_ul_he_mooney_rivlin.3.Omega(solid.kappa, v, u)
#                 + dw_ul_bulk_penalty.3.Omega( solid.K, v, u )
#                 = 0""",
#     }

solid = Material('solid', K=1e3, mu=20e0, kappa=10e0, rho=8000.0)
t1 = Term.new('dw_ul_he_neohook(solid.mu, v, u)', integral, omega, solid=solid, v=v, u=u)
t2 = Term.new('dw_ul_he_mooney_rivlin(solid.kappa, v, u)', integral, omega, solid=solid, v=v, u=u)
t3 = Term.new('dw_ul_bulk_penalty(solid.K, v, u)', integral, omega, solid=solid, v=v, u=u)
eq1 = Equation('balance of forces', t1 + t2 + t3)
eqs = Equations([eq1])

z_displacements = np.linspace(1e-1, 5e-1, 5)
vm_stresses = np.zeros([len(z_displacements), 2])
for i, z_displacement in enumerate(z_displacements):

    fix_bot = EssentialBC('fix_bot', bot, {'u.all': 0.0})
    fix_top = EssentialBC('fix_top', top, {'u.[0,1]': 0.0, 'u.[2]': -z_displacement})

    # ls = ScipyDirect({})
    #
    # nls_status = IndexedStruct()
    # nls = Newton({}, lin_solver=ls, status=nls_status)
    # # 'i_max': 1, 'eps_a': 1e-10
    #
    # pb = Problem('elasticity', equations=eqs)
    # pb.save_regions_as_groups('regions')
    #
    # pb.set_bcs(ebcs=Conditions([fix_bot, fix_top]))
    #
    # pb.set_solver(nls)
    #
    # status = IndexedStruct()
    # state = pb.solve(status=status)
    #
    # strain = pb.evaluate('ev_cauchy_strain.2.Omega(u)', u=u, mode='el_avg')
    # stress = pb.evaluate('ev_cauchy_stress.2.Omega(m.D, u)', m=m, u=u, mode='el_avg')
    # vms = get_von_mises_stress(stress.squeeze())
    # np.savetxt('tmp_vms.dat', vms)
    # vms = np.loadtxt('tmp_vms.dat')
    #
    # vol = mesh.cmesh.get_volumes(3)
    # np.savetxt('tmp_vol.dat', vol)
    # vol = np.loadtxt('tmp_vol.dat')
    #
    # vm_stresses[i, 0] = np.sum(vms * vol) / np.sum(vol)
    # vm_stresses[i, 1] = np.max(vms)
    #
    # pb.save_state('voronoi_foam_%f.vtk' % z_displacement, state)
    #
    ### Solvers ###
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton(
        {'i_max' : 20},
        lin_solver=ls, status=nls_status
    )

    ### Problem ###
    pb = Problem('hyper', equations=equations)
    pb.set_bcs(ebcs=ebcs)
    pb.set_ics(ics=Conditions([]))
    tss = SimpleTimeSteppingSolver(ts, nls=nls, context=pb)
    pb.set_solver(tss)

    ### Solution ###
    axial_stress = []
    axial_displacement = []
    def stress_strain_fun(*args, **kwargs):
        return stress_strain(
            *args, order=order, global_stress=axial_stress,
            global_displacement=axial_displacement, **kwargs)

    pb.solve(save_results=True, post_process_hook=stress_strain_fun)

show = True
if show:
    view = Viewer('voronoi_foam_%f.vtk' % z_displacement)
    view(vector_mode='warp_norm', rel_scaling=1,
         is_scalar_bar=True, is_wireframe=True)
