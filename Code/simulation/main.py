import socket
import json
import sys
import numpy as np
sys.path.append("D:\\TataElxsi\\Code")
import server

v_lst = []
id_lst = []

def make_vehicle(id):
    global v_lst
    global id_lst
    if id not in id_lst:
        v1 = server.vehicle(id)
        v_lst.append(v1)
        return v1

if __name__ == "__main__":
    file = "D:\\TataElxsi\\Code\\server\\DEM.txt"
    dem = server.process_dem.read_dem(file)
    dem = np.array(dem)
    obstacles = np.zeros(dem.shape)
    mp = server.map(dem, obstacles)
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 8888))
    server_socket.listen(1)
    print("Server listening...")

    # Accept connections from CoppeliaSim

    # Receive commands from CoppeliaSim and send predictions
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connected to {addr}")
        data = client_socket.recv(1024)
        data = data.decode('utf-8')
        data = json.loads(data)
        print(data)
        if data["id"] not in id_lst:
            v1 = make_vehicle(data['id'])
            id_lst.append(data["id"])
        
        v1 = v_lst[0]
        for v in v_lst:
            if v.ID == data["id"]:
                v1 = v
        # print(data["data"]["start_position"][0])
        # print(type(data["data"]["start_position"][0]))  
                      
        start_location = (int(data["data"]["start_position"][0]), int(data["data"]["start_position"][1]))
        end_location = (int(data["data"]["end_position"][0]), int(data["data"]["end_position"][1]))
        v1.location(start_location, end_location)
        print(v1.stating_location, v1.end_location)
        pth = v1.a_star(mp)
        print(pth)
        prediction_data = {'reply' : pth}
        prediction_data = json.dumps(prediction_data)
        prediction_data = prediction_data.encode('utf-8')
        client_socket.sendall(prediction_data)
        client_socket.close()

    # Close sockets
    server_socket.close()

