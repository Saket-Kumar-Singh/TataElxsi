    
from time import time, sleep
import casadi as ca
import numpy as np
from casadi import sin, cos, pi, tan
import matplotlib.pyplot as plt
from simulation_code import simulate


# Defi
L = 4 # Length of the vehicle (distance between front and rear wheels)
v_max = 5  # Maximum velocity
v_min = -5  # Minimum velocity
delta_max = pi  # Maximum steering angle
delta_min = -pi  # Minimum steering angle
N = 20
dt = 0.1  # Time step duration    
# Define symbolic variables for the states and controls
x = ca.SX.sym('x')  # x-position
y = ca.SX.sym('y')  # y-position
theta = ca.SX.sym('theta')  # orientation
v = ca.SX.sym('v')  # velocity
delta = ca.SX.sym('delta')  # steering angle
# phi = ca.sx.sym('phi')

x_init = 0
y_init = 20
theta_init = -pi/2

x_target = 15
y_target = 14
theta_target = -pi/4

Q_x = 1
Q_y = 1
Q_theta = 1
R1 = 3
R2 = 3



# Define the state vector
states = ca.vertcat(x, y, theta)

# Define the control vector
controls = ca.vertcat(v, delta)

n_states = states.numel()
n_controls = controls.numel()

X = ca.SX.sym('X', n_states, N+1)
U = ca.SX.sym('U', n_controls, N)

P = ca.SX.sym('P', n_states*(N+1) + n_controls)

Q = ca.diagcat(Q_x, Q_y, Q_theta)
R = ca.diagcat(R1, R2)


# Define the dynamics of the bicycle model
x_dot = ca.SX.sym('x_dot')
y_dot = ca.SX.sym('y_dot')
theta_dot = ca.SX.sym('theta_dot')


x_dot = v * cos(theta)
y_dot =  v * sin(theta)
theta_dot = v * sin(delta) / L

rhs = ca.vertcat(
    x_dot, 
    y_dot, 
    theta_dot,
)

def shift_timestep(dt, t0, state_init, u, f):
    f_value= state_init + ca.repmat(dt, n_states, 1)*f(state_init, u[:, 0])
    next_state= ca.DM.full((f_value))

    t0 = t0 + dt
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )
    # return
    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())


# Define the discrete-time dynamics using a forward Euler integration scheme
# rhs = states + change
f = ca.Function('f', [states, controls], [rhs])
# f is the dynamics of the system

# Define the initial and target states
# x_init = 15
# y_init = 10
# theta_init = -pi/4

# x_target = 0
# y_target = 0
# theta_target = 0


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

lbx[0:n_states*(N+1) : n_states] = -2000
lbx[1:n_states*(N+1) : n_states] = -2000
lbx[2:n_states*(N+1) : n_states] = -ca.inf

ubx[0:n_states*(N+1) : n_states] = 2000
ubx[1:n_states*(N+1) : n_states] = 2000
ubx[2:n_states*(N+1) : n_states] = ca.inf

lbx[n_states*(N+1) : :n_controls] = v_min
ubx[n_states*(N+1) : :n_controls] = v_max

lbx[n_states*(N+1) +1 : : n_controls] = -pi/2
ubx[n_states*(N+1) + 1: : n_controls] = pi/2
# lbx[0:n_states*(N+1) : n_states] = -20

# extrs = {"allow_free" : True}

# las_con = ca.SX.sym('las_con', n_controls, 1)
# Implement Range Kutta
for k in range(N):
    las_con = P[(N+1)*n_states:]
    # print(las_con)
    st = X[:, k]
    con = U[:, k]
    con_diff = con - las_con 
    # con_diff = ca.fabs(con_diff)

    cost_fn = cost_fn + (st - P[(k+1)*n_states:(k+2)*n_states]).T @ Q @ (st - P[(k+1)*n_states:(k+2)*n_states]) + con_diff.T @ R @ con_diff
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st+ (dt/2)*k1, con)
    k3 = f(st+ (dt/2)*k2, con)
    k4 = f(st + dt*k3, con)
    # print(st, k1)
    st_next_RK4 = st + ca.repmat(dt/6, n_states, 1)*(k1 + 2*k2 + 2*k3 +  k4)
    g = ca.vertcat(g, st_next - st_next_RK4)
    las_con = con



# print(k1)

# sleep(5)

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
    'ubg' : ca.DM.zeros((n_states*(N+1), 1)),
    'lbx' : lbx,
    'ubx' : ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])

t =ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))
X0 = ca.repmat(state_init, 1, N+1)

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([0])
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
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
# except:
#     print("Problem here")

# Define initial guess for the optimization variables
# initial_guess = np.zeros(N * (states.numel() + controls.numel()))
# param_values = [x_init, y_init, theta_init,  x_target, y_target, theta_target]
# solver_output = solver(x0=initial_guess, p=param_values)


if __name__ == '__main__':
    def func(t):
        if t <= 3:
            return [2*t, 10 - 2*t, -pi/4]
        else:
            return [6 + 2*(t - 3), 4 + 4*(t- 3), 1.107148721]
    main_loop = time()  # return time in sec
    las_con = ca.DM.zeros(n_controls, 1)
    sim_time = 30
    state_target = ca.DM([0, 3, -0.588233337])
    xs = []
    ys = []
    desx = []
    desy = []
    while (ca.norm_2(state_init - state_target) > 1e-5) and (mpc_iter * dt < sim_time):
        t1 = time()
        desx.append(func(mpc_iter*dt)[0])
        desy.append(func(mpc_iter*dt)[1])
        args['p'] = ca.vertcat(
            state_init,     # current state
        )
        for i in range(N):
            args['p'] = ca.vertcat(
            args['p'],
            ca.DM(func((mpc_iter + i)*dt))
        )
        args['p'] = ca.vertcat(
            args['p'],
            las_con,     # current state
        )
        # print(args['p'])
        # sleep(5)   
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))
        cat_controls = np.vstack((

            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(dt, t0, state_init, u, f)
        xs.append(state_init[0])
        ys.append(state_init[1])
        las_con= u0[:, 0]
        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))
        
        mpc_iter = mpc_iter + 1
        state_target = ca.DM(func(mpc_iter*dt))

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    tme = 0
    ind = 0
    # print(desx)
    tme_arr = []
    cat = []
    ind = 0
    # for i in range(int(len(cat_controls)/2)):
    #     cat.append(cat_controls[ind])
    #     ind+=2
    #     tme_arr.append(tme)
    #     tme = tme + dt
    #     plt.scatter(, tme)
    #     tme = dt + tme
    #     ind+=2
    plt.plot(xs, ys)
    plt.plot(desx, desy)
    plt.show()
    # simulate
    # simulate(cat_states, cat_controls, times, dt, N,
    #          np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)