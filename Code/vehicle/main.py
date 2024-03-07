
# import location 
# import math
# import socket 
# from communication import *

# THRESHHOLD = 20
# ID = 1

# def read_node():    
#     f = open("path.txt", "r")
#     p = f.read()
#     p = p.split(" ")
#     nodes = []
#     for k in p:
#         x = k.split(",")
#         t1 = int(x[0])
#         t2 = int(x[1])
#         nodes.append((t1, t2))

# def request_new_nodes(x, y):
#     print(f"will call new path starting from {x},{y}")
#     f = open("path.txt", "w")
#     f.write("0,0 1,1 2,2 3,3 4,4 5,5")
#     f.close()

# def dist(a, b, c, d):
#     return math.sqrt((a-c)**2 + (b - d)**2)

# def find_closest(x, y, nodes):
#     closest = 1e9+7
#     x1, y1
#     for p in nodes:
#         if dist(x, y, p[0], p[1]) < closest:
#             closest = dist(x, y, p[0], p[1])
#             x1 = p[0]
#             y1 = p[1]
#     print("The closest node is ")
#     if(closest >= THRESHHOLD):
#         return None        
#     return (x1, y1)

# import math

# def initialize_parameters():
#     Xk, Yk, θk = 0, 0, 0
#     Xg, Yg = 10, 10
#     v, Δβ, dmin, θmin, _, _, _, T = 1, 0.1, 1, math.pi/4, 0, 0, 0, 0.1
#     return Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin, T

# def reached_destination(Xk, Yk, Xg, Yg, T):
#     return (Xk - Xg)**2 < T**2 and (Yk - Yg)**2 < T**2

# def path_update(Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin):
#     θg = math.atan2(Yg - Yk, Xg - Xk)
#     β = θg - θk

#     x = Xk + v * math.cos(θk + β)
#     y = Yk + v * math.sin(θk + β)
#     θk = β

#     return x, y, θk

# def scan_for_intersection(Xk, Yk, Xobs, Yobs):
#     dobs = math.sqrt((Xk - Xobs)**2 + (Yk - Yobs)**2)
#     θobs = math.atan2(Yk - Yobs, Xk - Xobs)
#     return dobs, θobs

# def obstacle_detection(dobs, θobs, θg, dmin, θmin, Δβ):
#     θobs_g = θobs - θg

#     if (dobs < dmin) and (abs(θobs_g) < θmin):
#         if θobs_g > 0:
#             return Δβ
#         elif θobs_g < 0:
#             return -Δβ

#     return 0

# def update_reference_path(x, y, β, v):
#     x = x + v * math.cos(β)
#     y = y + v * math.sin(β)
#     return x, y

# def path_planning():
#     Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin, T = initialize_parameters()

#     while not reached_destination(Xk, Yk, Xg, Yg, T):
#         x, y, θk = path_update(Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin)
#         dobs, θobs = scan_for_intersection(x, y, 5, 5)  # Replace with actual obstacle coordinates
#         Δβ = obstacle_detection(dobs, θobs, θk, dmin, θmin, Δβ)
#         x, y = update_reference_path(x, y, θk + Δβ, v)
#         Xk, Yk = x, y

#     print("Destination reached.")

from mpc import mpc_solve
import matplotlib.pyplot as plt
from math import pi, atan, sin, cos
import casadi as ca
import numpy as np


def tan_inv(st1, st2):
    if(st1[0] == st2[0]):
        if(st1[1] > st2[1]):
            return pi/2
        else:
            return pi/2
    else:
        return atan((st1[1] - st2[1])/(st1[0] - st2[0]))      

if __name__ == "__main__":
        
    slver1 = mpc_solve()
    arr = [(0,0), (5, 5), (5, 10), (10, 10), (15, 15), (15, 10), (20, 10), (20, 5), (25,5), (25,10), (30, 15), (30, 20), (30, 30), (25, 25)]
    def func(t):
        if(t/2.35 >= len(arr) - 1):
            return [*arr[-1], 0]
        
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


    x = []
    y = []
    for t in arr:
        x.append(t[0])
        y.append(t[1])
    plt.plot(x, y)
    print("Ok")
    t = [0]
    for i in range(2000):
        t.append(t[-1] + 0.1)

    x_pos = []
    y_pos = []
    for tm in t:
        x_pos.append(func(tm)[0])
        y_pos.append(func(tm)[1])

    x_init = [0,0,0]
    t = 0
    las_con = [0,0]
    x_mpc = [0]
    y_mpc = [0]
    x0 = ca.repmat(x_init, 1, slver1.N+1)
    u = np.zeros((slver1.n_controls, slver1.N))
    las_con = u[:, 0]

    state_final = [arr[-1][0], arr[-1][1], tan_inv(arr[-1], arr[-2])]

    def dest_reached(state_init):
        return ca.norm_2((state_init[:, 0] - state_final)) < 1

    state_init = ca.DM([0, 0, 0])
    while not dest_reached(state_init) and t < 50:
        x0, u = slver1.mpc_control(x0,u, t, func, las_con)
        t_, state_init, u_ =  slver1.shift_timestep(0.1, t, state_init, u, slver1.f)
        u = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )
        las_con = u[:, 0]
        print(las_con)
        t = t + 0.1
        x_mpc.append(float(state_init[0][0]))
        y_mpc.append(float(state_init[1][0]))
        x0 = ca.horzcat(
            state_init, 
            x0[:, 2:],
            ca.reshape(x0[:, -1], -1, 1)
        )

    plt.scatter(10,10,s = (1**2) * 100, color = 'red', marker = 'o')   
    plt.plot(x_mpc, y_mpc)     
    plt.show()   