from __future__ import print_function, absolute_import, division

import os
import unittest

from openmdao.api import Problem, Group, pyOptSparseDriver, IndepVarComp
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.utils.lgl import lgl
from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE


def ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments=False,
                              use_boundary_constraints=False, compressed=False):
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    _, optimizer = set_pyoptsparse_opt(optimizer, fallback=False)
    p.driver.options['optimizer'] = optimizer
    p.driver.options['dynamic_simul_derivs'] = True
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 20
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings["Linesearch tolerance"] = 0.10
        p.driver.opt_settings['iSumm'] = 6
    if optimizer == 'SLSQP':
        p.driver.opt_settings['MAXIT'] = 50

    num_seg = 15
    seg_ends, _ = lgl(num_seg + 1)

    phase = dm.Phase(ode_class=AircraftODE,
                     transcription=dm.Radau(num_segments=num_seg, segment_ends=seg_ends,
                                            order=3, compressed=compressed,
                                            solve_segments=solve_segments))

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
    assumptions.add_output('S', val=427.8, units='m**2')
    assumptions.add_output('mass_empty', val=1.0, units='kg')
    assumptions.add_output('mass_payload', val=1.0, units='kg')

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0),
                           duration_bounds=(300, 10000),
                           duration_ref=5600)

    fix_final = True
    if use_boundary_constraints:
        fix_final = False
        phase.add_boundary_constraint('mass_fuel', loc='final', units='lbm',
                                      equals=1e-3, linear=False)
        phase.add_boundary_constraint('alt', loc='final', units='kft', equals=10.0, linear=False)

    phase.set_state_options('range', units='NM', fix_initial=True, fix_final=False, ref=1e-3,
                            defect_ref=1e-3, lower=0, upper=2000)
    phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=fix_final,
                            upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)
    phase.set_state_options('alt', units='kft', fix_initial=True, fix_final=fix_final, lower=0.0,
                            upper=60, ref=1e-3, defect_ref=1e-3)

    phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                      rate_continuity=True, rate2_continuity=False)

    phase.add_control('mach', units=None, opt=False)

    phase.add_input_parameter('S', units='m**2')
    phase.add_input_parameter('mass_empty', units='kg')
    phase.add_input_parameter('mass_payload', units='kg')

    phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0, shape=(1,))

    p.model.connect('assumptions.S', 'phase0.input_parameters:S')
    p.model.connect('assumptions.mass_empty', 'phase0.input_parameters:mass_empty')
    p.model.connect('assumptions.mass_payload', 'phase0.input_parameters:mass_payload')

    phase.add_objective('range', loc='final', ref=-1.0e-4)

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 3600.0
    p['phase0.states:range'][:] = phase.interpolate(ys=(0, 724.0), nodes='state_input')
    p['phase0.states:mass_fuel'][:] = phase.interpolate(ys=(30000, 1e-3), nodes='state_input')
    p['phase0.states:alt'][:] = 10.0

    p['phase0.controls:mach'][:] = 0.8

    p['assumptions.S'] = 427.8
    p['assumptions.mass_empty'] = 0.15E6
    p['assumptions.mass_payload'] = 84.02869 * 400

    p.run_driver()

    return p


class TestExSteadyAircraftFlight(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'test_ex_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_ex_aircraft_steady_flight_opt(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments=False)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)

    def test_ex_aircraft_steady_flight_solve(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments=True,
                                      use_boundary_constraints=True)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)


if __name__ == '__main__':
    unittest.main()
