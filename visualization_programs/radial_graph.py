
import pygame
from pygame.locals import *
import multiprocessing
import math as m
from pyomyo import Myo, emg_mode
import serial.tools.list_ports
import numpy as np

q = multiprocessing.Queue()

graph_color = (0,255,0)

# ------------ Functions for plotting ---------------

def rectRotated(scr, color, cr_pos, sides, alpha): 
    """
    cr_pos = [cx, cy, r]
    sides = [l1, l2]
    alpha in degrees
    """
    alpha = alpha/180*m.pi
    x = cr_pos[0]
    y = cr_pos[1]
    r = cr_pos[2]
    l1 = sides[0]
    l2 = sides[1]
    a = [int(x-r*m.sin(alpha)-l1/2*m.cos(alpha)), int(y-r*m.cos(alpha)+l1/2*m.sin(alpha))]
    b = [int(x-r*m.sin(alpha)+l1/2*m.cos(alpha)), int(y-r*m.cos(alpha)-l1/2*m.sin(alpha))]
    c = [int(x-r*m.sin(alpha)+l1/2*m.cos(alpha)-l2*m.sin(alpha)), int(y-r*m.cos(alpha)-l1/2*m.sin(alpha)-l2*m.cos(alpha))]
    d = [int(x-r*m.sin(alpha)-l1/2*m.cos(alpha)-l2*m.sin(alpha)), int(y-r*m.cos(alpha)+l1/2*m.sin(alpha)-l2*m.cos(alpha))]
    pygame.draw.polygon(scr,color,(a,b,c,d),0)

def show_text(scr, msg, font, x, y , color=(0,0,0)):
    text = font.render(msg.capitalize(), True, color)
    if text.get_width()>500:
        new_msg = msg.split(' ',1)
        text1 = font.render(new_msg[0].capitalize(), True, color)
        text2 = font.render(new_msg[1].capitalize(), True, color)
        text1_rect = text1.get_rect(center=(x,y))
        text2_rect = text2.get_rect(center=(x,y))
        scr.blit(text1, text1_rect)
        scr.blit(text2, text2_rect)
    else:
        text_rect = text.get_rect(center=(x, y))
        scr.blit(text, text_rect)

#--------------------------------------------------------------------------------

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
    print("\nSerial ports found: ")
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
    print("\nSelect baudrate: ")
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

def arduinoSerial(q, arduinoPort, baud,):

    # initialization variables
    old_packet = "0,0,0,0,0,0,0,0,0"
    old_vals = 0
    values = [0,0,0,0,0,0,0,0,0]
    old_values = [0,0,0,0,0,0,0,0,0]
    start = 0

    # preparing serial
    serialArduino = serial.Serial(arduinoPort, baud, timeout = None)
    rl = ReadLine(serialArduino)
    serialArduino.close()
    serialArduino.open()
    serialArduino.write(b"SON\n")    
    try:
        while True:
            if serialArduino.in_waiting:
                #start = time.perf_counter_ns()
                packet = rl.readline()
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
                        values[8] = int(packet_utf[32:])
                        old_values = values
                    #print(STATUS[4])
                    #print(packet_utf)
                except:
                    values = old_values
                # emg_reading format:
                # [0:9] Emg
                # [10] Repetition
                # [11] Pose
                # [12] Angle
                q.put(values)

    except KeyboardInterrupt:
        serialArduino.write(b"SOFF\n")
        serialArduino.close()
        

def graph_plot(scr, w,h, font1, vals, max_val):
    CHANNELS = len(vals)-1
    alpha = 150
    alpha = alpha/180*m.pi
    l = [3, 375]
    cx = w/2
    cy = h/2
    poly_pnt = []

    for i in range(CHANNELS):
        vals[i] = vals[i]*l[1]/max_val
    scr.fill((0,0,0))
    for i in range(CHANNELS):
        rectRotated(scr, (255,255,255), [cx, cy, 0], l, -180+i*360/CHANNELS)
        show_text(scr, str(i), font1, cx-(l[1]*1.5)*3/4*m.sin(i*2/CHANNELS*m.pi-m.pi), cy-(l[1]*1.5)*3/4*m.cos(i*2/CHANNELS*m.pi-m.pi), (255,255,255))
        poly_pnt.append((cx + vals[i]*m.sin(i*2/CHANNELS*m.pi), cy + vals[i]*m.cos(i*2/CHANNELS*m.pi)))
    show_text(scr, "Angle:", font1, w*0.9,0.05*h, (255,255,255))
    show_text(scr, f"{vals[-1]}", font1, w*0.9,0.1*h, (255,255,255))
    pygame.draw.polygon(scr, graph_color, poly_pnt)
    pygame.display.update()

if __name__ == "__main__":

    # Switch to True for testing UI, False to actually use the code
    test = False
    if not test:
        port, baud = choseArduinoSerial()
        p = multiprocessing.Process(target=arduinoSerial, args=(q,port, baud))
        p.start()

    # screen height and width
    h = 900
    w = int(h*4/3)

    pygame.init()
    scr = pygame.display.set_mode((w,h))
    pygame.display.set_caption("Radial Graphing")
    font_size = int(50)
    font1 = pygame.font.SysFont(None,font_size)
    clock = pygame.time.Clock()

    old_max = 0
    try:
        old_angle = 0
        while not test:
            #Get the emg data and plot it
            pygame.event.pump()
            clock.tick(60)
            while not(q.empty()):
                
                emg = list(q.get()) 
                new_max = max(emg[0:8])
                if new_max>old_max:
                    old_max = new_max
                graph_plot(scr, w,h, font1, emg, old_max)

        while test:
            pygame.event.pump()
            graph_plot(scr, w,h, font1, [100,200,300,500,800,1000,1200,1500,-120], 1500)
			
    except KeyboardInterrupt:
        print("Quitting")
        pygame.quit()
        p.terminate()
        p.join()
        exit()
