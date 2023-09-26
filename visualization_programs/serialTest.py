import serial
import time
import serial.tools.list_ports

import pygame
from pygame.locals import *
import multiprocessing

q = multiprocessing.Queue()

class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)


def choseArduinoSerial():
    baudList = [9600, 19200, 38400, 57600, 115200, 230400, 460800]
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No available port found.")
        exit()
    portList = []
    port = '0'
    i = 0
    print("Serial ports found: ")
    for onePort in ports:
        portList.append(str(onePort))
        print(str(i)+". "+str(onePort))
        i+=1
    val = input("Choose the Arduino Serial Port: ")
    if val == "":
        port = portList[0]
    else:
        port = portList[int(val)]
    porta = port.split(" ")
    port = porta[0]
    print(port + " selected")
    print("Select baudrate: ")
    for baudrate in baudList:
        print(str(i)+". "+str(baudrate))
        i+=1
    print("(Deafaul baudrate: 460800)")
    val = input()
    if val == "":
        baud = 460800
    else:
        baud = baudList[int(val)-1]
    print(f"{baud} baudrate selected")
    
    return port, baud

def SerialRead(q, port, baud):

    old_packet = "0"*44
    values = [0,0,0,0,0,0,0,0] #add zero for the angle
    old_values = [0,0,0,0,0,0,0,0]
    ser = serial.Serial(port, baud, timeout = None, bytesize=8, parity="N", stopbits=1)
    rl = ReadLine(ser)
    ser.close()
    ser.open()
    start = 0
    new_time = 0
    old_time = 0
    ser.flush()
    time.sleep(0.5)
    ser.write(b"SON")
    try:
        while True:
        # new_time = time.perf_counter_ns()
            
            if ser.in_waiting:
                start = time.perf_counter_ns()
                # Read the transmitted bytes   
                packet = rl.readline()      
                #print(packet_utf)
                #print(round((new_time-old_time)*pow(10,-9),4))
                #old_time = new_time
                try:
                    packet_utf = packet.decode('utf-8', errors='ignore').strip()
                    if len(packet_utf)==36:
                        values[0] = int(packet_utf[0:4])
                        values[1] = int(packet_utf[4:8])
                        values[2] = int(packet_utf[8:12])
                        values[3] = int(packet_utf[12:16])
                        values[4] = int(packet_utf[16:20])
                        values[5] = int(packet_utf[20:24])
                        values[6] = int(packet_utf[24:28])
                        values[7] = int(packet_utf[28:32])
                        #\values[8] = int(packet_utf[32:])
                        old_values = values
                    #print(received_str)
                except:
                    values = old_values

                end = time.perf_counter_ns()
                #print(f"{round((end-start)*pow(10,-9),4)}")
                #print(round((end-start)*pow(10,-9),4))
                #print(f"Received values: {values}")
                #end = start
                q.put(values)
    except KeyboardInterrupt:
        ser.write(b"SOFF")
        ser.close()
        print("Serial Closed")
        exit()

last_vals = None
def plot(scr, vals):
	DRAW_LINES = True

	global last_vals
	if last_vals is None:
		last_vals = vals
		return

	D = 5
	scr.scroll(-D)
	scr.fill((0, 0, 0), (w - D, 0, w, h))
	for i, (u, v) in enumerate(zip(last_vals, vals)):
		if DRAW_LINES:
			pygame.draw.line(scr, (0, 255, 0),
							 (w - D, int(h/9 * (i+1 - u/5))),
							 (w, int(h/9 * (i+1 - v/5))))
			pygame.draw.line(scr, (255, 255, 255),
							 (w - D, int(h/9 * (i+1))),
							 (w, int(h/9 * (i+1))))
		else:
			c = int(255 * max(0, min(1, v)))
			scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8))

	pygame.display.flip()
	last_vals = vals

if __name__ == "__main__":
    
    port, baud = choseArduinoSerial()
    serRead = multiprocessing.Process(target=SerialRead, args=(q,port, baud))
    serRead.start()

    w, h = 900, 800
    scr = pygame.display.set_mode((w, h))
    while q.empty():
         pass
    try:
        while True:
            # Handle pygame events to keep the window responding
            pygame.event.get()
            # Get the emg data and plot it
            while not(q.empty()):
                emg = list(q.get())
                plot(scr, [e / 500. for e in emg])
                #print(emg)

    except KeyboardInterrupt:
        print("Quitting")
        pygame.quit()
        serRead.close()
        exit()