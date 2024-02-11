import socket
import json
import requests

# IP address and port to listen on
listen_ip = '0.0.0.0'
listen_port = 12345

# Create a TCP socket
while(True):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to the listen address and port
        s.bind((listen_ip, listen_port))

        # Listen for incoming connections
        s.listen()

        print('Waiting for incoming connection...')

        # Accept incoming connection
        conn, addr = s.accept()

        with conn:
            print('Connected by', addr)

            # Receive data
            data = conn.recv(1024)
            # data = data.decode()
            data = json.loads(data)
            try:
                print("Recieved : " , data["data"])
            except:
                raise FileNotFoundError("No file")
            # print('Received:', data.decode())
