    
from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

mpc = None

def make_mpc():
    # Define parameters for the bicycle model
    L = 0.5  # Length of the vehicle (distance between front and rear wheels)
    v_max = 1  # Maximum velocity
    v_min = -1  # Minimum velocity
    delta_max = 0.5  # Maximum steering angle
    delta_min = -0.5  # Minimum steering angle
    N = 10
    dt = 0.1  # Time step duration    
    # Define symbolic variables for the states and controls
    x = ca.SX.sym('x')  # x-position
    y = ca.SX.sym('y')  # y-position
    theta = ca.SX.sym('theta')  # orientation
    v = ca.SX.sym('v')  # velocity
    delta = ca.SX.sym('delta')  # steering angle
    # phi = ca.sx.sym('phi')
    Q_x = 1
    Q_y = 1
    Q_theta = 0.3

    R1 = 0.3
    R2 = 0.3

    def shift_timestep(dt, t0, state_init, u, f):
        f_value= f(state_init, u[:, 0])
        next_state= ca.DM.full(state_init + (dt * f_value))

        t0 = t0 + dt
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return t0, next_state, u0
    
    def DM2Arr(dm):
        return np.array(dm.full())
    

    # Define the state vector
    states = ca.vertcat(x, y, theta)

    # Define the control vector
    controls = ca.vertcat(delta, v)

    n_states = states.numel()
    n_controls = states.numel()

    X = ca.SX.sym('X', n_states, N+1)
    U = ca.SX.sym('U', n_controls, N)

    P = ca.SX.sym('P', n_states +  n_states)

    Q = ca.diagcat(Q_x, Q_y, Q_theta)
    R = ca.diagcat(R1, R2)


    # Define the dynamics of the bicycle model
    x_dot = ca.SX.sym('x_dot')
    y_dot = ca.SX.sym('x_dot')
    theta_dot = ca.SX.sym('x_dot')


    x_dot = v * cos(delta + theta)
    y_dot = v * sin(delta + theta)
    theta_dot = v * sin(delta) / L

    change = ca.vertcat(
        x_dot, 
        y_dot, 
        theta_dot,
    )


    # Define the discrete-time dynamics using a forward Euler integration scheme
    rhs = states + change
    f = ca.Function('f', [states, controls], [rhs])
    # f is the dynamics of the system
    
    # Define the initial and target states
    x_init = 0
    y_init = 0
    theta_init = 0
       
    x_target = 15
    y_target = 10
    theta_target = pi/4

    cost_fn = 0
    g = X[:,0] - P[:n_states]
    
    # Define the lower and upper bounds for the states and controls
    lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
    ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

    # ubx = ca.vertcat(
    #     20,
    #     20,
    #     ca.inf,  # theta upper bound
    #     v_max,   # v upper bound
    #     delta_max  # delta upper bound
    # )
    
    lbx[0:n_states*(N+1) : n_states] = -20
    lbx[1:n_states*(N+1) : n_states] = -20
    lbx[2:n_states*(N+1) : n_states] = -ca.inf

    ubx[0:n_states*(N+1) : n_states] = 20
    ubx[1:n_states*(N+1) : n_states] = 20
    ubx[2:n_states*(N+1) : n_states] = ca.inf

    lbx[n_states*(N+1) : ] = v_min
    ubx[n_states*(N+1) : ] = v_max
    # lbx[0:n_states*(N+1) : n_states] = -20

    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        cost_fn = cost_fn + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) + con.T @ R @ con
        st_next = X[:, k+1]
        k1 = f(st, con)
        k2 = f(st, dt/2*k1, con)
        k3 = f(st, dt/2*k2, con)
        k4 = f(st + dt * k3, con)
        st_next_RK4 = st + (dt/6)*(k1 + 2*k2 + 2*k3, k4)
        g = ca.vertcat(g, st_next - st_next_RK4)


    OPT_variable = ca.vertcat(
        X.reshape((-1, 1)), 
        U.reshape((-1, 1))
    )

    nlp_prob = {
        'f' : cost_fn,
        'x' : OPT_variable,
        'g' : g,
        'p' : P
    }

    args = {
        'lbg' : ca.DM.zeros((n_states*(N+1), 1)),
        'ubg' : ca.DM.sezos((n_states*(N+1), 1)),
        'lbx' : lbx,
        'ubx' : ubx
    }

    t0 = 0
    state_init = ca.DM([x_init, y_init, theta_init])
    state_target = ca.DM([x_target,y_target, theta_target])

    t =ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))
    X0 = ca.repmat(state_init, 1, N+1)

    mpc_iter = 0
    cat_states = DM2Arr(X0)
    cat_controls = DM2Arr(u0[:, 0])
    times =
    # Set up the solver options
    opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }

    # Create the solver
    try:
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    except:
        print("Problem here")

    # Define initial guess for the optimization variables
    initial_guess = np.zeros(N * (states.numel() + controls.numel()))
    param_values = [x_init, y_init, theta_init,  x_target, y_target, theta_target]
    solver_output = solver(x0=initial_guess, p=param_values)
    return solver_output

if __name__ == "__main__":
    k = mpc()
    print(k, type(k))