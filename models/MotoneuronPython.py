from matplotlib import pyplot as plt
import numpy as np

gnabar = 0.05 # (mho/cm2)
gkleak = 0.002 # (mho/cm2)
gkrect = .3 # (mho/cm2)
gcaN = 0.05 # (mho/cm2)
gcaL = 0.0001 # (mho/cm2)
gcak = 0.3 # (mho/cm2)
ca0 = 2
ena = 50.0 # (mV)
ek = -80.0 # (mV)
eleak = -70.0 # (mV)
amA = 0.4
amB = 66
amC = 5
bmA = 0.4
bmB = 32
bmC = 5
R = 8.314472
F = 96485.34

class Motoneuron(): 
    """
    Model motoneuron soma, as described in McInytre et al. 2002.
    """

    class GatingVariable():
        """
        Channel-specfic gating variables, which dictate the channel kinematics and activation status.
        """

        def __init__(self, 
                     use_inf: bool, 
                     dt: float): 
            self._use_inf = use_inf
            self._dt = dt

        def set_initial_state(self): 
            if self._use_inf:
                self.state = self.inf
            else: 
                self.state = self.alpha/(self.alpha + self.beta)

        def update(self):
            if self._use_inf: 
                self.state += ((self.inf - self.state)/self.tau) * self._dt
            else: 
                self.state += (self.alpha * (1 - self.state) - self.beta * self.state) * self._dt


    def __init__(self, 
                 dt: float = 0.025, 
                 cm: float = 10): 
        # set starting voltage
        self._vm = -65

        # set constants
        self._dt = dt
        self._cm = cm

        # define and set gating variables
        self._m = self.GatingVariable(use_inf=False, dt=dt)
        self._h = self.GatingVariable(use_inf=True, dt=dt)
        self._update_fast_sodium_gating_variables()
        self._m.set_initial_state()
        self._h.set_initial_state()

        self._n = self.GatingVariable(use_inf=True, dt=dt)
        self._update_potassium_gating_variables()
        self._n.set_initial_state()
        

    def _update_potassium_gating_variables(self):
        """
        Update delayed-rectifier potassium current gating variable.
        """
        self._n.tau = 5/(np.exp((self._vm + 50)/40) + np.exp(-(self._vm + 50)/50))
        self._n.inf = 1/(1 + np.exp((self._vm + 65)/7))
        
    def _update_potassium_current(self):
        """
        Update delayed-rectifier potassium current. 
        """
        self._update_potassium_gating_variables()
        self._n.update()
        self._ikrect = gkrect * self._n.state**4 * (self._vm - ek)
    
    def _update_fast_sodium_gating_variables(self): 
        """
        Update fast sodium current gating variables.
        """
        self._m.alpha = (0.4 * (-(self._vm + 66)))/(np.exp(-(self._vm+66)/5) - 1)
        self._m.beta = (0.4 * (self._vm + 32))/(np.exp((self._vm+32)/5) - 1)
        
        self._h.tau = 30/(np.exp((self._vm + 60)/15) + np.exp((self._vm + 60)/16))
        self._h.inf = 1/(1 + np.exp((self._vm + 65)/7))

    def _update_fast_sodium_current(self):
        """
        Update fast sodium current. 
        """
        self._update_fast_sodium_gating_variables()
        self._m.update()
        self._h.update()
        self._ina = gnabar * self._m.state**3 * self._h.state * (self._vm - ena)

    def _update_leak_potassium_current(self): 
        self._ikleak = gkleak * (self._vm - eleak)

    def update(self, stimulus_current):
        # Update synaptic currents
        self._update_fast_sodium_current()
        self._update_potassium_current()
        self._update_leak_potassium_current()

        # Update membrane potential
        ion_currents_sum = self._ina + self._ikleak + self._ikrect # + self._ican + self._ical + self._ikca
        self._vm += (stimulus_current - ion_currents_sum) * self._dt / self._cm
        return self._vm


if __name__ == '__main__':
    mn = Motoneuron()
    vm = []
    for t in range(1000): 
        vm.append(mn.update(stimulus_current=50))
    
    plt.plot(vm)
    plt.show()