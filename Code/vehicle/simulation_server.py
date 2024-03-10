import socket
import json
import numpy as np
from mpc import mpc_solve, tan_inv
from math import cos, sin, atan, pi
import casadi as ca
from time import sleep
import math
from scipy.interpolate import interp1d

def return_message(client_socket, data):
    data = data.encode("utf-8")
    client_socket.sendall(data)

v_max = 0 
v_min = 0 
arr = []

# def func(t,  step_size = 5):
#     global v_max
#     global arr
#     # print("Path to be followed", arr)
#     # print("I will not go faster than", v_max)
#     v_m = v_max/2
#     __time = 1*step_size/v_m
#     print(t/__time)
#     if(t/__time >= len(arr) - 1):
#         return [*arr[-1], tan_inv(arr[-1], arr[-2])]
    
#     (x, y) = arr[int(t/__time)]
#     (x1, y1) = arr[int(t/__time) + 1]
#     p = int(t/__time)
#     # print(x, y)
#     # print(x1, y1)
#     theta = 0
#     if(x1 == x):
#         if(y1 > y):
#             theta = -pi/2
#         else:    
#             theta = pi/2
#     else:
#         theta = atan((y1 - y)/(x1 - x))
#     vx = v_m*cos(theta)
#     vy = v_m*sin(theta)
#     px = x + vx*(t - __time*p)
#     py = y + vy* (t - __time*p)
#     # To keep the value of p in bound
#     px = min(px, max(x,x1))
#     px = max(px, min(x, x1))
#     py = min(py, max(y,y1))
#     py = max(py, min(y,y1))
#     return [px, py, theta]

class traj():
    def __init__(self, arr, v_max):
        print("[SVR]Making Trajectory")
        self.arr = arr
        self.vm = v_max/1.5
        self.x = []
        self.y = []
        self.t = []
        self.x.append(arr[0][0])
        self.y.append(arr[0][1])
        self.t.append(0)
        t = 0
        for i in range(len(arr) - 1):
            t_now = t + math.sqrt((arr[i][0] - arr[i+1][0])**2 + (arr[i+1][1]- arr[i][1])**2)/self.vm
            self.x.append( arr[i+1][0])
            self.t.append(t_now)
            self.y.append (arr[i+1][1])
            t = t_now

        self.x_inter = interp1d(self.t, self.x, kind = "linear")
        self.y_inter = interp1d(self.t, self.y, kind = "linear")

    def func(self, t):
        x = self.x_inter(t)
        y = self.y_inter(t)
        # dy_dx = 0
        theta = 0
        if (self.x_inter(t+0.001) - self.x_inter(t)):
            dy_dx = (self.y_inter(t + 0.001) - self.y_inter(t))/(self.x_inter(t+0.001) - self.x_inter(t))
            theta = atan(dy_dx)
            if dy_dx < 0 and self.y_inter(t + 0.001) - self.y_inter(t) > 0:
                theta = pi + theta

        elif(self.y_inter(t + 0.001) > self.y_inter(t) ):
            theta = pi
        else:
            theta = -pi

        return [x, y, theta, self.vm, 0]


def main():
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 8800))
    server_socket.listen(1)
    print("Server listening...")

    # Accept connections from CoppeliaSim
    # client_socket, addr = server_socket.accept()
    # Receive commands from CoppeliaSim and send predictions
    client_socket, addr = server_socket.accept()
    print(f"Connected to {addr}")
    slver = ""
    while True:
        data = client_socket.recv(8000)
        ack = "ACK"
        client_socket.sendall(ack.encode("utf-8"))
        # print("recievedSomething")
        data = data.decode('utf-8')
        
        # print(data)
        if data:
            data = json.loads(data)
            # print("[RECIEVED]", data)
            if(data["id"] == "makeSolver"):
                print("Making Solver")
                # try:
                global v_max
                global arr
                global v_min
                v_max = data["data"]["v_max"]
                slver = mpc_solve(
                    data["data"]["x_init"],data["data"]["y_init"],\
                                   data["data"]["theta_init"], data["data"]["v_init"], data["data"]["v_max"],\
                                      data["data"]["v_min"], data["data"]["delta_max"], \
                                        data["data"]["delta_min"], data["data"]["N"]\
                                            , 0.005, data["data"]["L"], data["data"]["a_max"], data["data"]["a_min"])
                arr = data["data"]["arr"]
                v_max = data["data"]["v_max"]
                v_min = data["data"]["v_min"]
                data = {
                    "id" : "mpcMakingResult",
                    "data" : True
                }
                data = json.dumps(data)
                inter = traj(arr, v_max)
                return_message(client_socket, data)

            else:
                # print("No Shit")
                x0 = ca.DM(data["data"]["x0"])
                u = np.array(data["data"]["u"])
                t_now = data["data"]["t"]
                las_con = np.array(data["data"]["las_con"])
                x1 = np.array(data["data"]["x1"])
                x2 = np.array(data["data"]["x2"])
                x3 = np.array(data["data"]["x3"])
                x4 = np.array(data["data"]["x4"])
                x5 = np.array(data["data"]["x5"])
                x6 = np.array(data["data"]["x6"])
                x7 = np.array(data["data"]["x7"])
                x8 = np.array(data["data"]["x8"])
                # print(np.array(x0).shape)
                x0 = ca.DM(x0)
                u = np.array(u)
                x0, u = slver.mpc_control(x0, u, t_now, inter.func, las_con,x0, x1, x2, x3, x4, x5, x6, x7, x8)
                # print(u[:, 0])
                x0 = x0.full()
                x0 = x0.tolist()
                u = u.full()
                u = u.tolist()
                message = {
                    "id" : "controls",
                    "data" : {
                        "x0" : x0,
                        "u" : u 
                    } 
                }
                message = json.dumps(message)
                return_message(client_socket, message)
        sleep(0.1)
        # client_socket.close()
    # Close sockets
    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    main()