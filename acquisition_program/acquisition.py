import time
from time import sleep
import pygame
from pygame.locals import *
import multiprocessing
from multiprocessing import Lock
import numpy as np
import pandas as pd
import os
from pyomyo import Myo, emg_mode
import serial.tools.list_ports

#%% Global Variables

global main_color #bars' color
main_color = [0, 255, 0] #standard rgb

global myo_color #myo armband led colors
myo_color = [0, 128, 0] #rgb but max is 128 instead of 255

#%% Class for fast serial readLine
# Shout-out to skoehler for providing this class https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522

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

#%%     MYOband Acquisition Function

def saveCSV(myo_data, path):
        print("Finished collecting.")
        idx = np.where(myo_data==-1)[0][0]
        myo_data = myo_data[1:idx][:]
        print(f"{len(myo_data)} sample collected")
        # Add columns and save to df
        cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8", "Repetition", "Pose", "Angle"]
        myo_df = pd.DataFrame(myo_data, columns=cols)
        myo_df.to_csv(path, index=False)
        print(f"CSV Saved at: {path}")

def emg_worker(mode, filepath, folder, STATUS):
    try:
        m = Myo(mode=mode)
        m.connect()
        myo_data = []
        start = 0
        def add_to_data(emg, movement):
            myo_data.append(np.array(emg))
            myo_data[-1] = np.append(myo_data[-1], STATUS[2], axis=None)
            myo_data[-1] = np.append(myo_data[-1], STATUS[3], axis=None)
            myo_data[-1] = np.append(myo_data[-1], STATUS[4], axis=None)
            print(myo_data[-1])

        m.add_emg_handler(add_to_data)

        def print_battery(bat):
            print("Battery level:", bat)

        m.add_battery_handler(print_battery)

         # Its go time
        m.set_leds(myo_color, myo_color)
        # Vibrate to know we connected okay
        m.vibrate(1)
        #while STATUS[0] == 0 and STATUS[1] == 0: #wait untill START = 1
        #    myo_data = []
        print("Worker is Running")
        reset = True
        while True: #While Stop = False and Start = True
            m.run() 
            if STATUS[1] == 1 and reset:
                myo_data = []   # Deletes all the emg recording before the start signal
                print(myo_data)
                reset = False
            if STATUS[0] != 0:
                break
        print("Worker Stopped")
        if STATUS[0] == 1:
            saveCSV(myo_data, folder+filepath)
            exit()
    except KeyboardInterrupt:
        exit()

#%%     Arduino serial functions

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

def arduinoSerial(arduinoPort, baud, STATUS, myo, folder = '', filename = ''):

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
    
    if myo: #if using myo armband...
        try:
            while True:
                """
                while STATUS[0] == 0 and STATUS[1] == 0: #wait untill START = 1
                    #print("waiting")
                    STATUS[4] = 0
                """
                while STATUS[0] == 0:# and STATUS[1] == 1: #While Stop = False and Start = True
                    if serialArduino.in_waiting:
                        packet = rl.readline()
                        try:
                            packet_utf = packet.decode('utf-8', errors='ignore').strip()
                            if len(packet_utf)==36:
                                values[10] = int(packet_utf[32:])
                                old_values = values
                            #print(received_str)
                        except:
                            values = old_values
                        STATUS[4] = values[10]
                if STATUS[0] == 1:
                    serialArduino.write(b"SOFF\n")
                    serialArduino.close()
                    exit()

        except KeyboardInterrupt:
            serialArduino.close()
            exit()

    else:   #if reading directly from arduino...
        # In order to speed up the data processing I tried to pre-allocate the
        # array that will store the acquisition data (emg_reading).
        # I based the lenght on how much the acquisition would take and the
        # sampling frequency. To make the processing easier in matlab afterwards
        # I initialized all the array values to -1.
        Fs = 1000
        poseTime = 5
        restTime = 3
        poseNum = 10
        repNum = 6
        exTime = poseNum*repNum*poseTime + (poseNum*repNum+1)*restTime
        exSample = int(exTime*Fs*1.1)
        emg_reading = np.ones(13, np.int16)*-1
        myo_data = np.ones([exSample, 11], np.int16)*-1
        sample = 0
        emg_reading = np.zeros(11, np.int16)
        
        try:
            while True:
                #while STATUS[0] == 0 and STATUS[1] == 0 and STATUS[5] == 0: #wait untill START = 1
                    #print("waiting")
                 #   STATUS[4] = 0
                #  While Stop = False and (Start = True or Calibration = True)
                while STATUS[0] == 0:# and (STATUS[1] == 1 or STATUS[5] == 1): 
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
                                STATUS[4] = int(packet_utf[32:])
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
                        if STATUS[1] == 1:
                            np.put(emg_reading, [0,1,2,3,4,5,6,7], values[0:8])
                            np.put(emg_reading, [8, 9, 10], STATUS[2:])
                            myo_data[sample] = emg_reading
                            sample += 1
                            #print(emg_reading)
                            #print(f"{(time.perf_counter_ns()-start)*pow(10,-9)}")

                if STATUS[0] == 1:
                    serialArduino.write(b"SOFF\n")
                    saveCSV(myo_data, folder+filename)
                    serialArduino.close()
                    exit()

        except KeyboardInterrupt:
            serialArduino.write(b"SOFF\n")
            serialArduino.close()
            exit()


#%%     Visual Feedback Fuction

def screen_print(POSE_DURATION, REST_DURATION, POSE_REP, ANGLE_NUMBER, cal, poses, STATUS, train = True, modality = 1):
    
    #------ Pygame Window ---------------------------

    # all the dimansion SHOULD be proportional to h, so you SHOULD be able
    # to increase the window dimension without breaking anything

    h = 800
    w = int(h*4/3)
    font_size = int(min(w,h)*0.125)

    cx = w*2/3
    cy = h/2
    l1, l2 = min(w,h)*(3/4),min(w,h)*(3/4)
    lx, ly = cx-l1/2, cy-h*3/8     #Square Position

    if modality == 2:
        Tx, Ty = w/4, font_size/2
        Rx, Ry = w/2, font_size/2
        Ax, Ay = w*3/4-int(0.019*w), font_size/2
        ATx, ATy = w*3/4+int(w*0.16), font_size/2
    else:
        Tx, Ty = w/3, font_size/2
        Rx, Ry = w*2/3, font_size/2
        Ax, Ay = w/5, h/5
        ATx, ATy = w/5, h*3/10
    

    pygame.init()
    scr = pygame.display.set_mode((w,h))
    if train:
        pygame.display.set_caption("Training")
    else:
        pygame.display.set_caption("Acquisition")

    #------ Text Variables --------------------------

    font1 = pygame.font.SysFont(None,font_size)
    font2 = pygame.font.SysFont(None,int(font_size/2))

    #------ Pose Images -----------------------------
    
    POSE_IMG_PATH = "Pose/" #add final slash
    i1, i2 = min(l1,l2)*(3/5), min(l1,l2)*(3/5)
    ix, iy = cx-h*9/40, cy-h/8  #Image Position
    img = False
    imgs = os.listdir(POSE_IMG_PATH)
    pose_img_dict = {}
    # preload all images available with corresponding name
    for i in range(len(imgs)):
        imgs[i] = imgs[i].replace('.png','')
        pose_img = pygame.image.load(POSE_IMG_PATH+imgs[i]+'.png')
        pose_img = pygame.transform.scale(pose_img, (i1, i2))
        pose_img_dict[imgs[i]] = pose_img
       

    #----------------------------------------------
    clock = pygame.time.Clock()  
    POSE = False
    current_pose = 0
    pose_number = len(poses)
    pose_idx = 0
    angle_idx = 0
    rep_count = 0

    #------ Drawing Functions -----------------------

    def show_text(scr, msg, font, x, y, color=(0,0,0)):
        text = font.render(msg.capitalize(), True, color)
        if text.get_width()>500:
            new_msg = msg.split(' ', 1)
            text1 = font.render(new_msg[0].capitalize(), True, color)
            text2 = font.render(new_msg[1].capitalize(), True, color)
            text1_rect = text1.get_rect(center=(x, y-font_size/3))
            text2_rect = text2.get_rect(center=(x, y+font_size/3))
            scr.blit(text1, text1_rect)
            scr.blit(text2, text2_rect)
        else:
            text_rect = text.get_rect(center=(x, y)) # centers the text at the coordinates of origin
            scr.blit(text, text_rect)

    def draw_rect(scr,color, x, y, l1, l2):
        pygame.draw.rect(scr, color, (x,y,l1,l2))

    #------ Calibration Func ------------------------

    #def round_down(n, decimals=0):
    #    multiplier = 10 ** decimals
    #    return math.floor(n * multiplier) / multiplier

    def round_to_multiple(number, multiple):
        return multiple*round(number/multiple)

    def angleCalibration(scr, STATUS):
        print("Calibration...")
        STATUS[5] = 1
        min_angle = None
        max_angle  = None
        while min_angle == None or max_angle == None:
            scr.fill((0,0,0))
            show_text(scr, "Angle:", font1, w/2-int(0.075*w), h/2, (255, 255, 255))
            show_text(scr, f"{STATUS[4]}°", font1, w/2+int(w*0.125), h/2, (255, 255, 255))
            if min_angle == None:
                show_text(scr, "Set Min Angle", font1, w/2, h/4, (255, 255, 255))
            if max_angle == None and min_angle != None:
                show_text(scr, "Set Max Angle", font1, w/2, h/4, (255, 255, 255))

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if max_angle == None and min_angle != None:
                        max_angle = STATUS[4]
                    if min_angle == None:
                        min_angle = STATUS[4]
                    
            pygame.display.update()
        print(f"Min Angle {min_angle}°\tMax Angle {max_angle}°")
        angle_increment = (max_angle-min_angle)/4
        angles = [min_angle, round_to_multiple(angle_increment,5), round_to_multiple(angle_increment*2,5),round_to_multiple(angle_increment*3,5),max_angle]
        """
        # Old angle algorithm
        angle_increment = round_to_multiple(abs(max_angle)/(ANGLE_NUMBER-1),5)
        i=0
        while i*angle_increment<=abs(max_angle):
            angles.append(i*angle_increment)
            i+=1
        if len(angles)<ANGLE_NUMBER:
            angles.append(max_angle)
        """
        print(angles)
        STATUS[5] = 0
        return angles

    try:
        while True:
            pygame.event.get()

            # Calibration
            if STATUS[1] == 0 and (cal):
                angles = angleCalibration(scr, STATUS)
                
            while STATUS[1] == 0:
                # Waiting for the Start

                scr.fill((0,0,0))
                draw_rect(scr, (0,255,0), lx, ly, l1, l2)
                show_text(scr, "Press any key", font1, cx, cy-font_size/2)
                show_text(scr, "to start", font1, cx, cy+font_size/2)
                show_text(scr, "Angle:", font1, Ax, Ay, (255, 255, 255))
                show_text(scr, f"{STATUS[4]}°", font1, ATx, ATy, (255, 255, 255))
                if modality == 2:
                    # write pose list
                    i=0
                    for pose in poses:
                        show_text(scr, pose.replace("_"," "),font2, (cx-l1/2)/2, ly+font_size*i/2, color = (255,255,255))
                        i+=1
                pygame.display.update()


                for event in pygame.event.get():
                    if event.type == KEYDOWN:		#wait for a keydown event
                        STATUS[1] = 1  				#Start = True
                        start_time = time.time()
                        phase_time = start_time
                        print("GO!")

            # Main Loop

            while True:
                pygame.event.get()
                scr.fill((0, 0, 0))
                # if not all poses are done
                if pose_idx<pose_number:
                    if angle_idx<ANGLE_NUMBER:
                        if modality == 2:
                            # pose list on the left side, the name is green if it's the current pose
                            i = 0
                            for pose in poses:
                                if pose == poses[pose_idx]:
                                    color = main_color
                                else:
                                    color = (255,255,255)
                                show_text(scr, pose.replace("_"," "),font2, (cx-l1/2)/2, ly*1.02+font_size*i/2, color = color)
                                i+=1
                        # adding image if it's available
                        if poses[pose_idx] in imgs:
                            pose_img = pose_img_dict[poses[pose_idx]]
                            img = True
                        else:
                            img = False
                        # setting starting time
                        current_time = int(time.time()-phase_time)
                        # text to be displayed
                        time_text = f"Time: {current_time+1}"
                        rep_text = f"Rep: {rep_count+1}"
                        angle_text = f"{STATUS[4]}°"
                        show_text(scr, time_text, font1, Tx, Ty, (255, 255, 255))
                        show_text(scr, rep_text, font1, Rx, Ry, (255, 255, 255))
                        show_text(scr, "Angle:", font1, Ax, Ay, (255, 255, 255))
                        show_text(scr, angle_text, font1, ATx, ATy, (255, 255, 255))
                        if modality == 1:                        
                            show_text(scr, "Target", font1, Ax, Ay+h/3-font_size/2, (255, 255, 255))
                            show_text(scr, "Angle:", font1, Ax, Ay+h/3+font_size/2, (255, 255, 255))
                            show_text(scr, f"{angles[angle_idx]}°", font1, ATx, ATy+h*5/12, (255, 255, 255))

                        # if there are stil reps to be done:
                        if rep_count<POSE_REP:
                            # if it's POSE and not REST:
                            if POSE:
                                # if the phase time is less than the pose duration;
                                if ((time.time()-phase_time) <= POSE_DURATION):
                                    STATUS[2] = rep_count+1
                                    STATUS[3] = pose_idx+1
                                    draw_rect(scr, main_color, lx, ly, l1, l2)
                                    show_text(scr, poses[pose_idx].replace('_',' '), font1, cx, cy-l2/3)
                                    if img:
                                        scr.blit(pose_img, (ix, iy))
                                # Else change from POSE to REST and reset time
                                else:
                                    phase_time = time.time()
                                    POSE = not POSE
                                    rep_count += 1
                            # if it's REST and not POSE 
                            else:
                                # if the phase time is less than the rest duration;
                                if ((time.time()-phase_time) <= REST_DURATION):
                                    STATUS[2] = 0
                                    STATUS[3] = 0
                                    draw_rect(scr, (125,125,125), lx, ly, l1, l2)
                                    show_text(scr, 'Rest', font1, cx, cy-l2/3)
                                    if img:
                                        scr.blit(pose_img, (ix, iy))
                                # Else change from REST to POSE and reset time
                                else:
                                    phase_time = time.time()
                                    POSE = not POSE
                            print(f"Clock: {int(clock.get_fps())} fps", end="\r")
                            pygame.display.update()
                            clock.tick(120)
                        # if trial is finished (all the reps are done)
                        else:
                                rep_count = 0 # reset rep counter
                                if modality == 2:
                                    pose_idx += 1 # go to next pose
                                if modality == 1:
                                    angle_idx += 1
                    # if all angles are done
                    else:
                        phase_time = time.time()
                        break
                # if the poses are done reset time for last rest phase
                else:
                    phase_time = time.time()
                    break
            # last rest phase last longer in other to have cleaner
            # background noise samples
            while (time.time()-phase_time) <= REST_DURATION*2:
                pygame.event.get()
                scr.fill((0, 0, 0))
                current_time = int(time.time()-phase_time)
                time_text = f"Time: {current_time+1}"
                show_text(scr, time_text, font1, Tx, Ty, (255, 255, 255))
                STATUS[2] = 0
                STATUS[3] = 0
                draw_rect(scr, (125,125,125), lx, ly, l1, l2)
                show_text(scr, 'Rest', font1, cx, cy-l2/3)
                pygame.display.update()
                clock.tick(120)

            # The acquisition is compleated and all the samples are saved
            scr.fill((0, 0, 0))
            STATUS[0] = 1   #Stop = True
            STATUS[1] = 1
            draw_rect(scr, (0,255,0), lx, ly, l1, l2)
            show_text(scr, "Acquisition", font1, cx, cy-font_size/2)
            show_text(scr, "Completed", font1, cx, cy+font_size/2)
            pygame.display.update()
            sleep(2)
            pygame.quit()
            exit()
    except KeyboardInterrupt:
        pygame.quit()
        exit()

#%%     Main Function

if __name__ == "__main__":

    STATUS = multiprocessing.Array('i', 6, lock = False) #[STOP, START, REP, POSE, ANGLE, CALIBRATION STATUS]
    STATUS[0] = 0 #Stop
    STATUS[1] = 0 #Start
    STATUS[2] = 0 #Repetition
    STATUS[3] = 0 #Pose
    STATUS[4] = 0 #Angle from Arduino
    STATUS[5] = 0 #Calibration Status

    # array format:
    # [0:9] Emg
    # [10] Repetition
    # [11] Pose
    # [12] Angle

    msg = input("Training? (Y/n)")
    if msg == 'n' or msg == 'N':
        train = False
    else:
        train = True
    
    arduinoPort,baudrate = choseArduinoSerial()
    
    # Pose Variables
    POSE_DURATION = 5   # seconds
    REST_DURATION = 3   # seconds
    POSE_REP = 6        # number of repetition per pose 
    ANGLE_NUMBER = 5    # number of angles to acquire
    DEGREE = 1
    
    myo = False
    cal = False
    '''
     Currently the Angle Number is hardcoded as 5, since I changed the algorithm that calculates
     the acquisition angles, so changing the variable will only change the name of the file.
     I'll look into adding a more generalized algorithm later.

     If you want to do an acquisition for each pose with various angle just put one pose in the exercise
     and maybe manually change the file name. If you want to do an acquisition of serie of poses for a
     specific angle we can comment out the calibration part.
     Changing the 'modality' variable should let you select from 'one pose, multiple angles' (mode 1), and
     'series of pose, one angle' (mode 2). The default is mode 1
    '''
    modality = int(input("\nSelect Mode: \n1. one pose, multiple angles\n2. series of pose, one angle\n"))
    
    if modality == 2 and not train:
        ANGLE_NUMBER = 1
        m = input("\nDo you want to do the calibration? (y/N)")
        if m == 'y' or m == 'Y':
            cal = True
        else:
            cal = False
    else:
        cal  = False
    if not train:
        DEGREE = input("\nWhich angle are you recording? [1, 2, 3, 4, 5]\n")
        # True if using myo armband
        val = input("\nAre you using myoband for acquisition? (y/N)\n")
        if val == "y" or val == "Y":
            myo = True
        else:
            myo = False

    poses = ['medium_wrap', 'lateral', 'extension_type', 'tripod', 'power_sphere', 'power_disk', 'prismatic_pinch',
             'index_extension', 'thumb_adduction', 'prismatic_4_fingers', 'wave_in', 'wave_out', 'fist', 'open_hand']

    ex = [['wave_in', 'wave_out'],
        ['wave_in', 'wave_out', 'fist', 'open_hand'],
        ['index_extension', 'prismatic_pinch', 'medium_wrap', 'lateral', 'fist'],
        ['medium_wrap', 'lateral', 'extension_type', 'tripod', 'power_sphere', 'power_disk', 'prismatic_pinch',
        'index_extension', 'thumb_adduction', 'prismatic_four_fingers', 'wave_in', 'wave_out', 'fist', 'open_hand'],
        ['medium_wrap', 'lateral','power_sphere', 'power_disk', 'prismatic_pinch',
        'index_extension', 'wave_out', 'wave_in', 'fist', 'open_hand']]

    
    #Change destination folder and filename before starting
    DESTINATION = "DATA/sub_3_offset/" #add forward slash at the end
    file_name = f"Sub3_{POSE_REP}rep_{POSE_DURATION}sec_angle_{DEGREE}_ESP32_1000Hz.csv"
    
    if not train:
        print(f"\nTraining: {train}\r\nModality: {modality}\r\nCalibration: {cal}\r\nAngle: {DEGREE}\r\nFile Name: {file_name}\n")
        input()
        if not os.path.exists(DESTINATION):
            os.makedirs(DESTINATION)
    if myo:
        mode = emg_mode.RAW
        myo_read = multiprocessing.Process(target = emg_worker, args=(mode, file_name, DESTINATION, STATUS))
    ard_read = multiprocessing.Process(target = arduinoSerial, args=(arduinoPort, baudrate, STATUS, myo, DESTINATION, file_name))

    if not train and myo:
        myo_read.start()
    ard_read.start()
    
    screen_print(POSE_DURATION, REST_DURATION, POSE_REP, ANGLE_NUMBER, cal, ex[4], STATUS, train, modality)

    """
    To Do:
    Now the program needs to be restated every time an acquistion finishes.
    When using modality 2 it could be useful to ask if we want to continue with the next acquisiton for the next angle.
    """