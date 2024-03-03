import socket
import json
import numpy as np
from mpc import mpc_solve, tan_inv
from math import cos, sin, atan, pi
import casadi as ca
from time import sleep

def return_message(data):
    data = data.encode("utf-8")
    client_socket.sendall(data)

global v_max 
global v_min 
global arr

def func(t,  step_size = 5):
    global v_max
    __time = step_size/v_max
    if(t/__time >= len(arr) - 1):
        return [*arr[-1], tan_inv(arr[-1], arr[-2])]
    
    (x, y) = arr[int(t/__time)]
    (x1, y1) = arr[int(t/__time) + 1]
    p = int(t/__time)
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
    px = x + vx*(t - __time*p)
    py = y + vy* (t - __time*p)
    # To keep the value of p in bound
    px = min(px, max(x,x1))
    px = max(px, min(x, x1))
    py = min(py, max(y,y1))
    py = max(py, min(y,y1))
    return [px, py, theta]

if __name__ == "__main__":
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

    while True:
        data = client_socket.recv(4096)
        ack = "ACK"
        client_socket.sendall(ack.encode("utf-8"))
        # print("recievedSomething")
        data = data.decode('utf-8')
        # print(data)
        if data:
            data = json.loads(data)
            print("[RECIEVED]", data)
            if(data["id"] == "makeSolver"):
                try:
                    global vmax
                    v_max = data["data"]["v_max"]
                    slver = mpc_solve(
                        data["data"]["x_init"],data["data"]["y_init"],\
                                       data["data"]["theta_init"], data["data"]["v_max"],\
                                          data["data"]["v_min"], data["data"]["delta_max"], \
                                            data["data"]["delta_min"], data["data"]["N"]\
                                                )
                    arr = data["data"]["arr"]
                    data = {
                        "id" : "mpcMakingResult",
                        "data" : True
                    }
                    data = json.dumps(data)
                    return_message(data)
                except:
                    data = {
                        "id" : "mpcMakingResult",
                        "data" : False
                    }     
                    data = json.dumps(data)           
                    return_message(data)

            else:
                # print("No Shit")
                x0 = ca.DM(data["data"]["x0"])
                u = np.array(data["data"]["u"])
                t_now = data["data"]["t"]
                las_con = np.array(data["data"]["las_con"])
                print(np.array(x0).shape)
                x0 = ca.DM(x0)
                u = np.array(u)
                x0, u = slver.mpc_control(x0, u, t_now, func, las_con)
                print(u[:, 0])
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
                return_message(message)
        sleep(0.1)
        # client_socket.close()
    # Close sockets
    client_socket.close()
    server_socket.close()