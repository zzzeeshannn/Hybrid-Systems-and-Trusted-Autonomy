'''
Stanley Bak
Autopilot State-Machine Logic

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
'''

import abc
import numpy as np

from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.controlled_f16 import controlled_f16
from aerobench.util import Freezable, get_state_names, StateIndex

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, llc=None):

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.freeze_attrs()

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals. Override this one
        in subclasses.

        returns four values per aircraft: Nz, ps, Ny_r, throttle
        '''

        return

    def get_checked_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals and check them against ctrl limits
        '''

        rv = np.array(self.get_u_ref(t, x_f16), dtype=float)

        assert rv.size % 4 == 0, "get_u_ref should return Nz, ps, Ny_r, throttle for each aircraft"

        for i in range(rv.size //4):
            Nz, _ps, _Ny_r, _throttle = rv[4*i:4*(i+1)]

            l, u = self.llc.ctrlLimits.NzMin, self.llc.ctrlLimits.NzMax
            assert l <= Nz <= u, f"autopilot commanded invalid Nz ({Nz}). Not in range [{l}, {u}]"

        return rv

    def der_func(self, t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = self.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + self.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft, f"full state size: {full_state.size}, " + \
            f"vars per aircraft = {num_vars}, num_aircraft = {num_aircraft}"

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]

            #print(f".called der_func(aircraft={i}, t={t}, state={full_state}")

            alpha = state[StateIndex.ALPHA]
            if not -2 < alpha < 2:
                raise SimModelError(f"alpha ({alpha}) out of bounds")

            vel = state[StateIndex.VEL]
            # even going lower than 300 is probably not a good idea
            if not 200 <= vel <= 3000:
                raise SimModelError(f"velocity ({vel}) out of bounds")

            alt = state[StateIndex.ALT]
            if not -10000 < alt < 100000:
                raise SimModelError(f"altitude ({alt}) out of bounds")

            u_ref = u_refs[4*i:4*(i+1)]

            v2_integrators = False
            xd = controlled_f16(t, state, u_ref, self.llc, self.llc.model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

class SimModelError(RuntimeError):
    'simulation state went outside of what the model is capable of simulating'
