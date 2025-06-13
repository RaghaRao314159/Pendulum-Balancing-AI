"""
fork from python-rl and pybrain for visualization
"""
#import numpy as np
import numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
import jax.numpy as jnp
import jax

# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
     
def remap_angle(theta):
    return _remap_angle(theta)
    
def _remap_angle(theta):
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta

def remap_angle_j(theta):
    return _remap_angle_j(theta)

def _remap_angle_j(theta):
    """Remap angle to be within [-pi, pi] using jax"""
    # account for fact that theta can be more than multiple of 2pi 
    theta = theta % (2. * jnp.pi)
    theta = jax.lax.cond(theta < -jnp.pi, lambda x: x + 2. * jnp.pi, lambda x: x, theta)
    theta = jax.lax.cond(theta > jnp.pi, lambda x: x - 2. * jnp.pi, lambda x: x, theta)
    return theta
    

## loss function given a state vector. the elements of the state vector are
## [cart location, cart velocity, pole angle, pole angular velocity]
def _loss(state):
    sig = 0.5
    return 1-np.exp(-np.dot(state,state)/(2.0 * sig**2))

def loss(state):
    return _loss(state)

# jax version of euler_integration
def eulerStep_j(state, force, params):
    # state: [cart_location, cart_velocity, pole_angle, pole_velocity]
    cart_location, cart_velocity, pole_angle, pole_velocity = state
    cart_mass, pole_mass, pole_length, mu_c, mu_p, gravity, max_force, delta_time = params
    s = jnp.sin(pole_angle)
    c = jnp.cos(pole_angle)
    m = 4.0*(cart_mass+pole_mass)-3.0*pole_mass*(c**2)

    cart_accel = (2.0*(pole_length*pole_mass*(pole_velocity**2)*s+2.0*(force-mu_c*cart_velocity))\
        -3.0*pole_mass*gravity*c*s + 6.0*mu_p*pole_velocity*c/pole_length)/m

    pole_accel = (-3.0*c*(2.0/pole_length)*(pole_length/2.0*pole_mass*(pole_velocity**2)*s + force-mu_c*cart_velocity)+\
        6.0*(cart_mass+pole_mass)/(pole_mass*pole_length)*\
        (pole_mass*gravity*s - 2.0/pole_length*mu_p*pole_velocity) \
        )/m

    # Update state variables
    dt = delta_time
    # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not.
    cart_velocity += dt * cart_accel
    pole_velocity += dt * pole_accel
    pole_angle    += dt * pole_velocity
    cart_location += dt * cart_velocity

    return jnp.array([cart_location, cart_velocity, pole_angle, pole_velocity]), None  # No carry value needed for this scan step

g = 9.8  # gravity
mass_cart = 0.5
mass_pole = 0.5
mu_c = 0.001
mu_p = 0.001
l = 0.5  # pole length
max_force = 5.0
delta_time = 0.1
sim_steps = 50
dt = delta_time / sim_steps

@jax.jit
def remap_angle2(theta):
    return jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi

@jax.jit
def cartpole_dynamics(state, force, max_force=max_force):
    force = max_force * jnp.tanh(force/max_force)
    x, x_dot, theta, theta_dot = state

    s = jnp.sin(theta)
    c = jnp.cos(theta)
    m = 4.0 * (mass_cart + mass_pole) - 3.0 * mass_pole * c**2

    cart_accel = (2.0 * (l * mass_pole * theta_dot**2 * s + 2.0 * (force - mu_c * x_dot)) -
                  3.0 * mass_pole * g * c * s + 6.0 * mu_p * theta_dot * c / l) / m

    pole_accel = (-3.0 * c * (2.0 / l) * (l / 2.0 * mass_pole * theta_dot**2 * s + force - mu_c * x_dot) +
                  6.0 * (mass_cart + mass_pole) / (mass_pole * l) *
                  (mass_pole * g * s - 2.0 / l * mu_p * theta_dot)) / m

    x_dot_new = x_dot + dt * cart_accel
    theta_dot_new = theta_dot + dt * pole_accel
    theta_new = theta + dt * theta_dot_new
    x_new = x + dt * x_dot_new

    return jnp.array([x_new, x_dot_new, remap_angle2(theta_new), theta_dot_new])
    # return jnp.array([x_new, x_dot_new, remap_angle2(theta_new), theta_dot_new])
    
@jax.jit
def cartpole_step(state, force, max_force=max_force):
    def body_fn(i, state):
        return cartpole_dynamics(state, force, max_force=max_force)

    return jax.lax.fori_loop(0, sim_steps, body_fn, state)


class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False, max_force=20):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi    # angle is defined to be zero when the pole is upright, pi when hanging vertically down
        self.pole_angle_j = jnp.pi
        self.pole_velocity = 0.0
        self.visual = visual

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.001 #   # friction coefficient of the cart
        self.mu_p = 0.001 # # friction coefficient of the pole
        self.sim_steps = 50         # number of Euler integration steps to perform in one go
        # self.sim_steps = 5000
        self.delta_time = 0.1       # time step of the Euler integrator
        # self.delta_time = 0.001       # time step of the Euler integrator
        self.max_force = max_force
        self.gravity = 9.8
        self.cart_mass = 0.5

        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2

        if self.visual:
            self.drawPlot()

    def setState(self, state):
        self.cart_location = state[0]
        self.cart_velocity = state[1]
        self.pole_angle = state[2]
        self.pole_velocity = state[3]
            
    def getState(self):
        return np.array(
            [self.cart_location,
             self.cart_velocity,
             self.pole_angle,
             self.pole_velocity]
        )
    
    def getState_j(self):
        return jnp.array(
            [self.cart_location,
             self.cart_velocity,
             self.pole_angle,
             self.pole_velocity]
        )

    # reset the state vector to the initial state (down-hanging pole)
    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi
        self.pole_velocity = 0.0
    
    def reset_j(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = jnp.pi
        self.pole_velocity = 0.0

    # This is where the equations of motion are implemented
    def performAction(self, action = 0.0):
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)

        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):
            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
            
            cart_accel = (2.0*(self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+2.0*(force-self.mu_c*self.cart_velocity))\
                -3.0*self.pole_mass*self.gravity*c*s + 6.0*self.mu_p*self.pole_velocity*c/self.pole_length)/m
            
            pole_accel = (-3.0*c*(2.0/self.pole_length)*(self.pole_length/2.0*self.pole_mass*(self.pole_velocity**2)*s + force-self.mu_c*self.cart_velocity)+\
                6.0*(self.cart_mass+self.pole_mass)/(self.pole_mass*self.pole_length)*\
                (self.pole_mass*self.gravity*s - 2.0/self.pole_length*self.mu_p*self.pole_velocity) \
                )/m

            # Update state variables
            dt = (self.delta_time / float(self.sim_steps))
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_velocity += dt * pole_accel
            self.pole_angle    += dt * self.pole_velocity
            self.cart_location += dt * self.cart_velocity

        if self.visual:
            self._render()
        
    def performAction_noise(self, action, std):
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)

        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):
            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
            
            cart_accel = (2.0*(self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+2.0*(force-self.mu_c*self.cart_velocity))\
                -3.0*self.pole_mass*self.gravity*c*s + 6.0*self.mu_p*self.pole_velocity*c/self.pole_length)/m
            
            pole_accel = (-3.0*c*(2.0/self.pole_length)*(self.pole_length/2.0*self.pole_mass*(self.pole_velocity**2)*s + force-self.mu_c*self.cart_velocity)+\
                6.0*(self.cart_mass+self.pole_mass)/(self.pole_mass*self.pole_length)*\
                (self.pole_mass*self.gravity*s - 2.0/self.pole_length*self.mu_p*self.pole_velocity) \
                )/m

            # Update state variables
            dt = (self.delta_time / float(self.sim_steps))
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_velocity += dt * pole_accel
            self.pole_angle    += dt * self.pole_velocity
            self.cart_location += dt * self.cart_velocity

        noise = np.random.normal(0, std, 4)
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle    += noise[2]
        self.pole_velocity += noise[3]

        if self.visual:
            self._render()

    # This is where the equations of motion are implemented
    def performAction_j(self, action = 0.0):
        # prevent the force from being too large
        force = self.max_force * jnp.tanh(action/self.max_force)

        params = (
            self.cart_mass,
            self.pole_mass,
            self.pole_length,
            self.mu_c,
            self.mu_p,
            self.gravity,
            self.max_force,
            self.delta_time / float(self.sim_steps),
        )
        state = jnp.array([self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity])

        def scan_step(state, _):
            return eulerStep_j(state, force, params)

        # Run scan for sim_steps
        state, _ = jax.lax.scan(scan_step, state, None, length=self.sim_steps)

        # Update the object's state
        self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity = state[0], state[1], state[2], state[3]

    # remapping as a member function
    def remap_angle(self):
        self.pole_angle = _remap_angle(self.pole_angle)

    def remap_angle_j(self):
        self.pole_angle = _remap_angle_j(self.pole_angle)

    # the loss function that the policy will try to optimise (lower) as a member function
    def loss(self):
        return _loss(self.getState())
    

   # the following are graphics routines
    def drawPlot(self):
        ion()
        self.fig = plt.figure()
        # draw cart
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.box = Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle)], 
                           [0, np.cos(self.pole_angle)], linewidth=3, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-2, 2)
        


    def _render(self):
        self.box.set_x(self.cart_location - self.cartwidth / 2.0)
        self.pole.set_xdata([self.cart_location, self.cart_location + np.sin(self.pole_angle)])
        self.pole.set_ydata([0, np.cos(self.pole_angle)])
        self.fig.show()
        
        plt.pause(0.1)

    # functions added by me
    # set simulation parameters
    def setSimParams(self, sim_steps=50, delta_time=0.1):
        self.sim_steps = sim_steps
        self.delta_time = delta_time

    # close the plot
    def close_plot(self):
        plt.close(self.fig)


