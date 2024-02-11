import serial.tools.list_ports
import time
ports= serial.tools.list_ports.comports()
serialInst = serial.Serial()


def make_connection():
    serialInst.baudrate = 9600
    serialInst.port = "/dev/ttyACM0"
    serialInst.open()



def loca():
    message = ""
    if (not serialInst) or (not serialInst.isOpen()):
        make_connection()
    p = 3
    message = ()
    while len(message) < 3:
        if serialInst.in_waiting:
            packet = serialInst.readline()
            # print(packet.decode('utf'))
            try:
                message = packet.decode('utf')
                message = message.split(',')
            except:
                message = ()   

    return message    
