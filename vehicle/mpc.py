    
from time import time, sleep
import casadi as ca
import numpy as np
from casadi import sin, cos, pi, tan, atan
import matplotlib.pyplot as plt
from simulation_code import simulate


# Defi
class mpc_solve():
    def __init__(self):
        self.L = 1 # Length of the vehicle (distance between front and rear wheels)
        v_max = 5  # Maximum velocity
        v_min = -5  # Minimum velocity
        delta_max = pi  # Maximum steering angle
        delta_min = -pi  # Minimum steering angle
        self.N = 20
        self.dt = 0.1  # Time step duration    
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
        self.states = ca.vertcat(x, y, theta)

        # Define the control vector
        self.controls = ca.vertcat(v, delta)

        self.n_states = self.states.numel()
        self.n_controls = self.controls.numel()

        X = ca.SX.sym('X', self.n_states, self.N+1)
        U = ca.SX.sym('U', self.n_controls, self.N)

        P = ca.SX.sym('P', self.n_states*(self.N+1) + self.n_controls)

        Q = ca.diagcat(Q_x, Q_y, Q_theta)
        R = ca.diagcat(R1, R2)


        # Define the dynamics of the bicycle model
        x_dot = ca.SX.sym('x_dot')
        y_dot = ca.SX.sym('y_dot')
        theta_dot = ca.SX.sym('theta_dot')


        x_dot = v * cos(theta)
        y_dot =  v * sin(theta)
        theta_dot = v * sin(delta) / self.L

        rhs = ca.vertcat(
            x_dot, 
            y_dot, 
            theta_dot,
        )




        # Define the discrete-time dynamics using a forward Euler integration scheme
        # rhs = states + change
        self.f = ca.Function('f', [self.states, self.controls], [rhs])
        # f is the dynamics of the system

        # Define the initial and target states
        # x_init = 15
        # y_init = 10
        # theta_init = -pi/4

        # x_target = 0
        # y_target = 0
        # theta_target = 0


        cost_fn = 0
        g = X[:,0] - P[:self.n_states]

        # Define the lower and upper bounds for the states and controls
        lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))

        # ubx = ca.vertcat(
        #     20,
        #     20,
        #     ca.inf,  # theta upper bound
        #     v_max,   # v upper bound
        #     delta_max  # delta upper bound
        # )

        lbx[0:self.n_states*(self.N+1) : self.n_states] = -2000
        lbx[1:self.n_states*(self.N+1) : self.n_states] = -2000
        lbx[2:self.n_states*(self.N+1) : self.n_states] = -ca.inf

        ubx[0:self.n_states*(self.N+1) : self.n_states] = 2000
        ubx[1:self.n_states*(self.N+1) : self.n_states] = 2000
        ubx[2:self.n_states*(self.N+1) : self.n_states] = ca.inf

        lbx[self.n_states*(self.N+1) : :self.n_controls] = v_min
        ubx[self.n_states*(self.N+1) : :self.n_controls] = v_max

        lbx[self.n_states*(self.N+1) +1 : : self.n_controls] = -pi/2
        ubx[self.n_states*(self.N+1) + 1: : self.n_controls] = pi/2
        # lbx[0:self.n_states*(self.N+1) : self.n_states] = -20

        # extrs = {"allow_free" : True}

        # las_con = ca.SX.sym('las_con', self.n_controls, 1)
        # Implement Range Kutta
        for k in range(self.N):
            las_con = P[(self.N+1)*self.n_states:]
            # print(las_con)
            st = X[:, k]
            con = U[:, k]
            con_diff = con - las_con 
            # con_diff = ca.fabs(con_diff)

            cost_fn = cost_fn + (st - P[(k+1)*self.n_states:(k+2)*self.n_states]).T @ Q @ (st - P[(k+1)*self.n_states:(k+2)*self.n_states]) + con_diff.T @ R @ con_diff
            st_next = X[:, k+1]
            k1 = self.f(st, con)
            k2 = self.f(st+ (self.dt/2)*k1, con)
            k3 = self.f(st+ (self.dt/2)*k2, con)
            k4 = self.f(st + self.dt*k3, con)
            # print(st, k1)
            st_next_RK4 = st + ca.repmat(self.dt/6, self.n_states, 1)*(k1 + 2*k2 + 2*k3 +  k4)
            g = ca.vertcat(g, st_next - st_next_RK4)
            las_con = con



        # print(k1)

        # sleep(5)

        self.OPT_variable = ca.vertcat(
            X.reshape((-1, 1)), 
            U.reshape((-1, 1))
        )

        self.nlp_prob = {
            'f' : cost_fn,
            'x' : self.OPT_variable,
            'g' : g,
            'p' : P
        }

        self.args = {
            'lbg' : ca.DM.zeros((self.n_states*(self.N+1), 1)),
            'ubg' : ca.DM.zeros((self.n_states*(self.N+1), 1)),
            'lbx' : lbx,
            'ubx' : ubx
        }

        t0 = 0
        state_init = ca.DM([x_init, y_init, theta_init])

        t = ca.DM(t0)

        u0 = ca.DM.zeros((self.n_controls, self.N))
        x0 = ca.repmat(state_init, 1, self.N+1)        
        mpc_iter = 0
        cat_states = self.DM2Arr(x0)
        cat_controls = self.DM2Arr(u0[:, 0])
        times = np.array([0])
        # Set up the solver options
        self.opts = {
            'ipopt': {
                'max_iter': 4000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

    def shift_timestep(self, dt, t0, state_init, u, f):
        f = self.f
        f_value= state_init + ca.repmat(self.dt, self.n_states, 1)*f(state_init, u[:, 0]) + ca.DM(0.01*np.random.randn(3, 1))
        next_state= ca.DM.full((f_value))

        t0 = t0 + self.dt
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )
           # return
        return t0, next_state, u0

    def DM2Arr(self, dm):
        return np.array(dm.full())

    def mpc_control(self,state_init,u,t_now, func, las_con):
        # slver1 = self
        # u0 = ca.DM.zeros((slver1.n_controls, slver1.N))
        # x0 = ca.repmat(state_init, 1, slver1.N+1)        
        self.args['p'] = ca.vertcat(
            state_init[:, 0],     # current state
        )
        for i in range(self.N):
            self.args['p'] = ca.vertcat(
            self.args['p'],
            ca.DM(func(t_now + (i+1)*0.1))
        )
        self.args['p'] = ca.vertcat(
            self.args['p'],
            las_con,     # current state
        )            
        self.args['x0'] = ca.vertcat(
            ca.reshape(state_init, self.n_states*(self.N+1), 1),
            ca.reshape(u, self.n_controls*self.N, 1)
        )
        sol = self.solver(
            x0 = self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        u = ca.reshape(sol['x'][self.n_states * (self.N + 1):],   self.n_controls, self.N)
        x0 = ca.reshape(sol['x'][: self.n_states * (self.N + 1)], self.n_states,   self.N+1)
        # new_con= u0[:, 0]
        return x0,u



if __name__ == '__main__':
    arr = [(0,0), (5, 5), (5, 10), (10, 10), (15, 15), (15, 10), (20, 10), (20, 5), (25,5), (25,10), (30, 15), (30, 20), (30, 30), (25, 25)]

    def func(t):
        if(t/2.35 >= len(arr) - 1):
            return [25, 25, -pi/4]
        
        (x, y) = arr[int(t/2.35)]
        (x1, y1) = arr[int(t/2.35) + 1]
        p = int(t/2.35)
        theta = 0
        if(x1 == x):
            if(y1 > y):
                theta = pi/2
            else:    
                theta = -pi/2
        else:
            theta = atan((y1 - y)/(x1 - x))
        vx = 3*cos(theta)
        vy = 3*sin(theta)
        px = x + vx*(t - 2.35*p)
        py = y + vy* (t - 2.35*p)

        # To keep the value of p in bound
        px = min(px, max(x,x1))
        px = max(px, min(x, x1))
        py = min(py, max(y,y1))
        py = max(py, min(y,y1))


        return [px, py, theta]
    
    t0 = 0
    slver1 = mpc_solve()    
    main_loop = time()  # return time in sec
    las_con = ca.DM.zeros(slver1.n_controls, 1)
    sim_time = 30
    times = 0

    xs = []
    ys = []
    desx = []
    desy = []
    x_init = 0
    y_init = 0
    theta_init = 0
    mpc_iter = 0
    state_init = ca.DM([x_init, y_init, theta_init])
    # x_target = 25
    # y_target = 25
    state_target = ca.DM([*arr[-1], pi/4])
    # theta_target = -pi/4
    conts = []
    u0 = ca.DM.zeros((slver1.n_controls, slver1.N))
    x0 = ca.repmat(state_init, 1, slver1.N+1)
    # cat_states = ca.DM([x_init, y_init, theta_init])
    # cat_controls = ca.DM([0,0])
    cat_states = slver1.DM2Arr(x0)
    cat_controls = slver1.DM2Arr(u0[:, 0])
    while (ca.norm_2(state_init - state_target) > 1e-5) and (mpc_iter * slver1.dt < sim_time):
        t1 = time()
        desx.append(func(mpc_iter*slver1.dt)[0])
        desy.append(func(mpc_iter*slver1.dt)[1])
        slver1.args['p'] = ca.vertcat(
            state_init,     # current state
        )
        for i in range(slver1.N):
            slver1.args['p'] = ca.vertcat(
            slver1.args['p'],
            ca.DM(func((mpc_iter + i)*slver1.dt))
        )
        slver1.args['p'] = ca.vertcat(
            slver1.args['p'],
            las_con,     # current state
        )
        # print(args['p'])
        # sleep(5)   
        # optimization variable current state
        slver1.args['x0'] = ca.vertcat(
            ca.reshape(x0, slver1.n_states*(slver1.N+1), 1),
            ca.reshape(u0, slver1.n_controls*slver1.N, 1)
        )
        sol = slver1.solver(
            x0 = slver1.args['x0'],
            lbx=slver1.args['lbx'],
            ubx=slver1.args['ubx'],
            lbg=slver1.args['lbg'],
            ubg=slver1.args['ubg'],
            p=slver1.args['p']
        )

        u = ca.reshape(sol['x'][slver1.n_states * (slver1.N + 1):], slver1.n_controls, slver1.N)
        x0 = ca.reshape(sol['x'][: slver1.n_states * (slver1.N + 1)], slver1.n_states, slver1.N+1)

        cat_states = np.dstack((
            cat_states,
            slver1.DM2Arr(x0)
        ))
        cat_controls = np.vstack((

            cat_controls,
            slver1.DM2Arr(u[:, 0])
        ))

        # t = np.vstack((
        #     t,
        #     t0
        # ))

        t0, state_init, u0 = slver1.shift_timestep(slver1.dt, t0, state_init, u, slver1.f)
        xs.append(state_init[0])
        ys.append(state_init[1])
        las_con= u0[:, 0]
        conts.append(int(las_con[1]))
        # print(slver1.x0)
        x0 = ca.horzcat(
            x0[:, 1:],
            ca.reshape(x0[:, -1], -1, 1)
        )
        print(x0.shape)
        # sleep(10)
        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))        
        mpc_iter = mpc_iter + 1
        state_target = ca.DM(func(mpc_iter*slver1.dt))

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
    #     tme = tme + self.dt
    #     plt.scatter(, tme)
    #     tme = self.dt + tme
    #     ind+=2

    # print(type(conts[0]))
    t = [0]
    fig, ax = plt.subplots()
    for i in range(int(sim_time/slver1.dt)-1):
        t.append(t[-1] + 0.1)
    fin_pos_x = []
    fin_pos_y = []
    for tm in t:
        fin_pos_x.append((func(tm))[0])
        fin_pos_y.append((func(tm))[1])
    # print(conts[0][0,0])    
    plt.plot(xs, ys)
    plt.scatter(fin_pos_x, fin_pos_y, s = 10)
    plt.show()
    # simulate
    simulate(cat_states, cat_controls, times, slver1.dt, slver1.N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),  ax, fig, save=False,)