import location 
import math
import socket 
from communication import *

THRESHHOLD = 20
ID = 1

def read_node():    
    f = open("path.txt", "r")
    p = f.read()
    p = p.split(" ")
    nodes = []
    for k in p:
        x = k.split(",")
        t1 = int(x[0])
        t2 = int(x[1])
        nodes.append((t1, t2))

def request_new_nodes(x, y):
    print(f"will call new path starting from {x},{y}")
    f = open("path.txt", "w")
    f.write("0,0 1,1 2,2 3,3 4,4 5,5")
    f.close()

def dist(a, b, c, d):
    return math.sqrt((a-c)**2 + (b - d)**2)

def find_closest(x, y, nodes):
    closest = 1e9+7
    x1, y1
    for p in nodes:
        if dist(x, y, p[0], p[1]) < closest:
            closest = dist(x, y, p[0], p[1])
            x1 = p[0]
            y1 = p[1]
    print("The closest node is ")
    if(closest >= THRESHHOLD):
        return None        
    return (x1, y1)

import math

def initialize_parameters():
    Xk, Yk, θk = 0, 0, 0
    Xg, Yg = 10, 10
    v, Δβ, dmin, θmin, _, _, _, T = 1, 0.1, 1, math.pi/4, 0, 0, 0, 0.1
    return Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin, T

def reached_destination(Xk, Yk, Xg, Yg, T):
    return (Xk - Xg)**2 < T**2 and (Yk - Yg)**2 < T**2

def path_update(Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin):
    θg = math.atan2(Yg - Yk, Xg - Xk)
    β = θg - θk

    x = Xk + v * math.cos(θk + β)
    y = Yk + v * math.sin(θk + β)
    θk = β

    return x, y, θk

def scan_for_intersection(Xk, Yk, Xobs, Yobs):
    dobs = math.sqrt((Xk - Xobs)**2 + (Yk - Yobs)**2)
    θobs = math.atan2(Yk - Yobs, Xk - Xobs)
    return dobs, θobs

def obstacle_detection(dobs, θobs, θg, dmin, θmin, Δβ):
    θobs_g = θobs - θg

    if (dobs < dmin) and (abs(θobs_g) < θmin):
        if θobs_g > 0:
            return Δβ
        elif θobs_g < 0:
            return -Δβ

    return 0

def update_reference_path(x, y, β, v):
    x = x + v * math.cos(β)
    y = y + v * math.sin(β)
    return x, y

def path_planning():
    Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin, T = initialize_parameters()

    while not reached_destination(Xk, Yk, Xg, Yg, T):
        x, y, θk = path_update(Xk, Yk, θk, Xg, Yg, v, Δβ, dmin, θmin)
        dobs, θobs = scan_for_intersection(x, y, 5, 5)  # Replace with actual obstacle coordinates
        Δβ = obstacle_detection(dobs, θobs, θk, dmin, θmin, Δβ)
        x, y = update_reference_path(x, y, θk + Δβ, v)
        Xk, Yk = x, y

    print("Destination reached.")



if __name__ == "__main__":
    k = socket.gethostname()
    if not check_present(k):
        k = make_vehicle(k)
        
    # print(get_end_point(k))

    path_planning()

    # exec()
    # while(True):
    #     x,y = get_location()
    #     try:
    #         nodes = read_node()
    #     except:
    #         request_new_nodes(x, y)
    #         nodes = read_node()

    #     k = find_closest(x, y, nodes)
    #     if not k:
    #         request_new_nodes(x, y)
    #         nodes  = read_node()    
    #         x1, y1 = x, y
    #     else:
    #         x1, y1 = k[0], k[1]
        
    #     ind = 0
    #     while(dist(x1, y1, ))       