# import yaml

# # Python data
# data = {
#     'server': {
#         'hostname': 'example.com',
#         'port': 8080,
#         'ssl_enabled': True
#     },
#     'database': {
#         'name': 'my_database',
#         'username': 'user123',
#         'password': 'secret123'
#     }
# }

# # Serialize Python data to YAML
# yaml_data = yaml.dump(data)

# print(yaml_data)


import socket
import time
import json
from get_location import loca
# IP address and port of the sink node
sink_ip = '192.168.178.100'
sink_port = 12345

s = ""

def make_conn():
    global s
    global sink_ip
    global sink_port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((sink_ip, sink_port))
    print("connection made")

def conn(data):
    global s
    print("Conn made")
    data = data.encode('utf-8')
    try:
        s.sendall(data)
    except:
        make_conn()
        s.sendall(data)    
    print('Data sent successfully.')
    time.sleep(0.1)