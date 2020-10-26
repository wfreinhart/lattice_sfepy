from __future__ import absolute_import
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.mechanics.tensors import get_von_mises_stress


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct

    ev = pb.evaluate
    stress = ev('ev_cauchy_stress.2.Omega(solid.D, u)', mode='el_avg')
    out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                  data=stress, dofs=None)

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)
    out['von_mises_stress'] = Struct(name='output_data', mode='cell',
                                     data=vms, dofs=None)

    strain = ev('ev_cauchy_strain.2.Omega(u)', mode='el_avg', verbose=False)
    out['cauchy_strain'] = Struct(name='output_data', mode='cell',
                                  data=strain, dofs=None)

    return out


filename_mesh = 'meshes/strut_test.vtk'
z_displacement = -0.10

regions = {
    'Omega' : 'all',
    'Top' : ('vertices in (z > 1.0)', 'facet'),
    'Bot' : ('vertices in (z < 0.0)', 'facet'),
}

materials = {
    'solid' : ({'D': stiffness_from_lame(dim=3, lam=1e1, mu=1e0)},),
}

fields = {
    'displacement': ('real', 'vector', 'Omega', 1),
}

integrals = {
    'i' : 1,
}

variables = {
    'u' : ('unknown field', 'displacement', 0),
    'v' : ('test field', 'displacement', 'u'),
}

ebcs = {
    'Fixed' : ('Bot', {'u.all' : 0.0}),
    'Displaced' : ('Top', {'u.[0,1]' : 0.0, 'u.[2]' : z_displacement}),
}

equations = {
    'balance_of_forces' :
    """dw_lin_elastic.i.Omega(solid.D, v, u) = 0""",
}

solvers = {
    'ls': ('ls.auto_direct', {}),
    'newton': ('nls.newton', {
        'i_max'      : 1,
        'eps_a'      : 1e-10,
    }),
}

options = {
    'post_process_hook' : 'post_process',
}

# run the model with the command:
# python run_sfepy.py linear_elasticity.py
# postprocess to view the von mises stress with this command:
# python run_postproc.py strut_test.vtk -b --only-names=u -d"u,plot_displacements,rel_scaling=1,color_kind='scalars',color_name='von_mises_stress'"
