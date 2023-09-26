'''
Use this program to plot in a circular 
histogram the values of the EMG sensors
'''

import pygame
from pygame.locals import *
import multiprocessing
import math as m
from pyomyo import Myo, emg_mode
import serial.tools.list_ports

#------------ Global Variables ---------------

global main_color #bars' color
main_color = [0, 255, 0] #standard rgb

global myo_color #myo armband led colors
myo_color = [0, 0, 128] #rgb but max is 128 instead of 255

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

# ------------ Myo Setup ---------------

q = multiprocessing.Queue()

def worker(q):
    m = Myo(mode=emg_mode.RAW)
    m.connect()
	
    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)
	
    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

     # Orange logo and bar LEDs
    m.set_leds(myo_color, myo_color)
    # Vibrate to know we connected okay
    m.vibrate(1)
	
    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")

# ---------- Serial Functions ------------

# Choose the arduino serial sort
def choseArduinoSerial():
    ports = serial.tools.list_ports.comports()
    portList = []
    i = 0
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
    return(port)

# Read the chosen serial port
def arduinoSerial(serialInst):
    if serialInst.in_waiting>0:
            packet = serialInst.readline()
            try:
                packet = packet.decode('utf')
                packet = packet[0:-2] # to remove the /r/n
            except:
                packet = old_packet
            old_packet = packet
            vals = packet.split(',')
            return vals[-1]
            # if using arduino to read the emg signals
            # just return the whole list vals

# ------------ Plot Function -------------

def round_plot(scr,w,h,font1,vals,angle):

    # To Do: modify this function when using arduino to also read emg sensors
    # EMG and angle are a single list: [emg, angle]

    r = 150
    l1 = 50
    l2 = 256

    CHANNELS = len(vals)
    alpha = 150
    alpha = alpha/180*m.pi

    logo = pygame.image.load("Logo Myoarmband.png")
    logo_size = 80
    logo = pygame.transform.scale(logo, (logo_size, logo_size))


    try:    
        scr.fill((0,0,0))
        for i in range(0,CHANNELS):
            # graph bars
            rectRotated(scr,(255,255,255), [h/2, h/2,r], [5, 256], -i*360/CHANNELS)
            rectRotated(scr,(255,255,255), [h/2, h/2,r], [30, 3], -i*360/CHANNELS)
            rectRotated(scr,(255,255,255), [h/2, h/2,r+128], [20, 3], -i*360/CHANNELS)
            rectRotated(scr,(255,255,255), [h/2, h/2,r+256], [30, 3], -i*360/CHANNELS)
            # emg value bars
            rectRotated(scr,main_color, [h/2, h/2, r], [l1, int(abs(vals[i]))], -i*360/CHANNELS)
            # electrode number
            show_text(scr, str(i+1), font1, h/2-r*3/4*m.sin(-i*360/CHANNELS/180*m.pi), h/2-r*3/4*m.cos(-i*360/CHANNELS/180*m.pi), (255,255,255))
            # marking electrode with the light
            if i==3:
                scr.blit(logo, (h/2-int(logo_size/2)-480*m.sin(-i*360/CHANNELS/180*m.pi), h/2-int(logo_size/2)-480*m.cos(-i*360/CHANNELS/180*m.pi)))
                #pygame.draw.circle(scr, myo_color, (h/2-450*m.sin(i*360/CHANNELS/180*m.pi), h/2-450*m.cos(i*360/CHANNELS/180*m.pi)), 10)
        pygame.draw.rect(scr, main_color, (875, 50, 250, 150), 5,2)
        show_text(scr, "Wrist Angle", font1, 1000, 100, (255,255,255))
        show_text(scr, f"{angle}°", font1, 1000, 150, (255,255,255))
        pygame.display.update()
    except KeyboardInterrupt:
        exit()


# -------- Main Program Loop -----------

if __name__ == "__main__":

    # Switch to True for testing UI, False to actually use the code
    test = False
    if not test:
        porta = choseArduinoSerial()
        serialArduino = serial.Serial()
        serialArduino.boudrate = 9600
        serialArduino.port = porta
        serialArduino.open()
        p = multiprocessing.Process(target=worker, args=(q,))
        p.start()

    # screen height and width
    h = 900
    w = int(h*4/3)

    pygame.init()
    scr = pygame.display.set_mode((w,h))
    font_size = int(50)
    font1 = pygame.font.SysFont(None,font_size)
    
    try:
        old_angle = 0
        while not test:
            #Get the emg data and plot it
            pygame.event.pump()
            if not test:
                angle = arduinoSerial(serialArduino)
            if angle is None:
                angle = old_angle
            while not(q.empty()):
                emg = list(q.get()) 
                round_plot(scr, w,h, font1, emg, angle)
                old_angle = angle
        while test:
            pygame.event.pump()
            round_plot(scr, w,h, font1, [100,200,300,500,800,1000,1200,1500], 0)
			
    except KeyboardInterrupt:
        print("Quitting")
        pygame.quit()
        quit()
