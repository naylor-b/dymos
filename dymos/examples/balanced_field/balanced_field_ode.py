import numpy as np
import openmdao.api as om


class BalancedFieldODEComp(om.ExplicitComponent):
    """
    The ODE System for an aircraft takeoff climb.

    Computes the rates for states v (true airspeed) gam (flight path angle) r (range) and h (altitude).

    References
    ----------
    .. [1] Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')
        self.options.declare('mode', values=('runway', 'climb'), desc='mode of operation (ground roll or flight)')

    def setup(self):
        nn = self.options['num_nodes']

        # Scalar (constant) inputs
        self.add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
        self.add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('T', val=1.0, desc='thrust', units='N')

        # Dynamic inputs (can assume a different value at every node)
        self.add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
        self.add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
        self.add_input('h', shape=(nn,), desc='altitude', units='m')
        self.add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')

        # Outputs
        self.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
        self.add_output('L', shape=(nn,), desc='lift force', units='N')
        self.add_output('D', shape=(nn,), desc='drag force', units='N')
        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
        self.add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2',
                        tags=['dymos.state_rate_source:v'])
        self.add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s',
                        tags=['dymos.state_rate_source:r'])
        self.add_output('W', shape=(nn,), desc='aircraft weight', units='N')
        self.add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)

        # Mode-dependent IO
        if self.options['mode'] == 'runway':
            self.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)
        else:
            self.add_input('gam', shape=(nn,), desc='flight path angle', units='rad')
            self.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle',
                            units='rad/s', tags=['dymos.state_rate_source:gam'])
            self.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s',
                            tags=['dymos.state_rate_source:h'])

        self.declare_coloring(wrt='*', method='cs')
        # self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = self.options['g']

        # Compute factor k to include ground effect on lift
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']
        m = inputs['m']
        T = inputs['T']
        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        e = inputs['e']

        outputs['W'] = W = m * g
        outputs['v_stall'] = v_stall = np.sqrt(2 * W / rho / S / CL_max)
        outputs['v_over_v_stall'] = v / v_stall

        outputs['CL'] = CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
        K_nom = 1.0 / (np.pi * AR * e)
        b = span / 2.0
        fact = ((h + h_w) / b) ** 1.5
        outputs['K'] = K = K_nom * 33 * fact / (1.0 + 33 * fact)

        outputs['q'] = q = 0.5 * rho * v ** 2
        outputs['L'] = L = q * S * CL
        outputs['D'] = D = q * S * (CD0 + K * CL ** 2)

        # Compute the downward force on the landing gear
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        # Runway normal force
        outputs['F_r'] = F_r = m * g - L * calpha - T * salpha

        # Compute the dynamics
        if self.options['mode'] == 'climb':
            gam = inputs['gam']
            cgam = np.cos(gam)
            sgam = np.sin(gam)
            outputs['v_dot'] = (T * calpha - D) / m - g * sgam
            outputs['gam_dot'] = (T * salpha + L) / (m * v) - (g / v) * cgam
            outputs['h_dot'] = v * sgam
            outputs['r_dot'] = v * cgam
        else:
            outputs['v_dot'] = (T * calpha - D - F_r * inputs['mu_r']) / m
            outputs['r_dot'] = v


import os
func_method = os.environ.get('USE_FUNC_COMP', '')

# # FIXME: remove after debugging
# func_method = 'jax'

if func_method in ('fd', 'cs', 'jax'):

    use_jax = func_method == 'jax'
    import openmdao.func_api as omf


    class BalancedFieldODEComp(om.ExplicitFuncComp):
        def __init__(self, **kwargs):
            nn = kwargs['num_nodes']
            isclimb = not ('mode' in kwargs and kwargs['mode'] == 'runway')

            if isclimb:
                def compute(rho, S, CD0, CL0, CL_max, alpha_max, h_w, AR, e, span, T, m, v, h, alpha, gam):
                    g = 9.80665
                    W = m * g
                    v_stall = np.sqrt(2 * W / rho / S / CL_max)
                    v_over_v_stall = v / v_stall

                    CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
                    K_nom = 1.0 / (np.pi * AR * e)
                    b = span / 2.0
                    fact = ((h + h_w) / b) ** 1.5
                    K = K_nom * 33 * fact / (1.0 + 33 * fact)

                    q = 0.5 * rho * v ** 2
                    L = q * S * CL
                    D = q * S * (CD0 + K * CL ** 2)

                    # Compute the downward force on the landing gear
                    calpha = np.cos(alpha)
                    salpha = np.sin(alpha)

                    # Runway normal force
                    F_r = m * g - L * calpha - T * salpha

                    # Compute the dynamics
                    cgam = np.cos(gam)
                    sgam = np.sin(gam)
                    v_dot = (T * calpha - D) / m - g * sgam
                    gam_dot = (T * salpha + L) / (m * v) - (g / v) * cgam
                    h_dot = v * sgam
                    r_dot = v * cgam

                    return CL, q, L, D, K, F_r, v_dot, r_dot, W, v_stall, v_over_v_stall, gam_dot, h_dot

            else:  # mode == runway

                def compute(rho, S, CD0, CL0, CL_max, alpha_max, h_w, AR, e, span, T, m, v, h, alpha, mu_r):
                    g = 9.80665
                    W = m * g
                    v_stall = np.sqrt(2 * W / rho / S / CL_max)
                    v_over_v_stall = v / v_stall

                    CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
                    K_nom = 1.0 / (np.pi * AR * e)
                    b = span / 2.0
                    fact = ((h + h_w) / b) ** 1.5
                    K = K_nom * 33 * fact / (1.0 + 33 * fact)

                    q = 0.5 * rho * v ** 2
                    L = q * S * CL
                    D = q * S * (CD0 + K * CL ** 2)

                    # Compute the downward force on the landing gear
                    calpha = np.cos(alpha)
                    salpha = np.sin(alpha)

                    # Runway normal force
                    F_r = m * g - L * calpha - T * salpha

                    # Compute the dynamics
                    v_dot = (T * calpha - D - F_r * mu_r) / m
                    r_dot = v

                    return CL, q, L, D, K, F_r, v_dot, r_dot, W, v_stall, v_over_v_stall


            f = (omf.wrap(compute)
                    .add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3')
                    .add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
                    .add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
                    .add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
                    .add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
                    .add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad')
                    .add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
                    .add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
                    .add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
                    .add_input('span', val=35.7, desc='Wingspan', units='m')
                    .add_input('T', val=1.0, desc='thrust', units='N')

                    # Dynamic inputs (can assume a different value at every node)
                    .add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
                    .add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
                    .add_input('h', shape=(nn,), desc='altitude', units='m')
                    .add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')
                    )

            if isclimb:
                f.add_input('gam', shape=(nn,), desc='flight path angle', units='rad')
            else:
                f.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)

            f = (f.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
                    .add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
                    .add_output('L', shape=(nn,), desc='lift force', units='N')
                    .add_output('D', shape=(nn,), desc='drag force', units='N')
                    .add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
                    .add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
                    .add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2',
                                    tags=['dymos.state_rate_source:v'])
                    .add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s',
                                    tags=['dymos.state_rate_source:r'])
                    .add_output('W', shape=(nn,), desc='aircraft weight', units='N')
                    .add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
                    .add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)
                    )

            if isclimb:
                f.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle',
                                    units='rad/s', tags=['dymos.state_rate_source:gam'])
                f.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s',
                                    tags=['dymos.state_rate_source:h'])

            # f.declare_option('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')

            # if func_method in ('fd', 'cs'):
            f.declare_coloring('*', method=func_method)
            f.declare_partials(of='*', wrt='*', method=func_method)

            skip = {'g', 'mode'}
            kwargs = {n: v for n, v in kwargs.items() if n not in skip}
            if use_jax and 'use_jit' not in kwargs:
                kwargs['use_jit'] = True
            super().__init__(f, **kwargs)

        def initialize(self):
            self.options.declare('num_nodes', types=int, default=1)
