import pygame, pygame_widgets, sys, os
import pandas as pd

from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame.locals import *
import math as m
from time import sleep

def load_dataset(folder_path):
    files = os.listdir(folder_path)
    new_files = []
    #get file names
    for file in files:
        if file.endswith('.csv'):
            new_files.append(file)

    #reorder file names by angle
    angles = ['']*len(new_files)
    for file in new_files:
        angle = file.split('_')
        angle = int(angle[1])-1
        angles[angle] = file
    db_by_angle = []
    for file in angles:
        db_by_angle.append(pd.read_csv(folder_path + file)) #angles x samples x (channels + rep + stim)
    cols = list(db_by_angle[0].columns) #colums
    m = len(db_by_angle[0])
    for i in range(0,len(db_by_angle)):
        if len(db_by_angle[i])<=m:
            m = len(db_by_angle[i])
            idx = i
    #print(idx)
    
    #cut dataframes the same size of the smallest one
    idxs = pd.Index([n for n in range(m)])
    new_db_by_angle = []
    for db in db_by_angle:
        a = db.tail(m)
        a.index = idxs
        new_db_by_angle.append(a)

    return new_db_by_angle

def rectRotated(scr, color, cr_pos, sides, alpha): 
    """
    cr_pos = [cx, cy, r]
    sides = [l1, l2]
    alpha in degrees
    """    
    alpha = alpha/180*m.pi
    r = cr_pos[2]
    x = cr_pos[0]
    y = cr_pos[1]
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

def main(db_by_angle):
    cols = list(db_by_angle[0].columns) #colums
    h = 900
    w = int(h*16/9)
    CHANNELS = 8
    alpha = 150
    alpha = alpha/180*m.pi
    r = 150
    l1 = 50
    l2 = 256
    pygame.init()
    scr = pygame.display.set_mode((w,h))
    font_size = int(50)
    font1 = pygame.font.SysFont(None,font_size)
    sliderTime = Slider(scr, 5/6*h+w/6-10,150,20,700, min = 0, max = len(db_by_angle[0])-1, step=1, color = (255,0,0), handleColour = (0,255,0), vertical =True)
    sliderAngle = Slider(scr, h/2+w/2-10,150,20,700, min = 0, max = 28, step=1, color = (255,0,0), handleColour = (0,255,0), vertical =True)
    sliderTime.value = 0
    sliderAngle.value = 0
    try:    
        while True:
            scr.fill((0,0,0))
            sample = sliderTime.getValue()
            angle = sliderAngle.getValue()
            current_db = db_by_angle[angle] #samples x (channels + rep + stim)


            for i in range(0,CHANNELS):
                rectRotated(scr,(255,255,255), [h/2, h/2,r], [5, 256], -i*360/8)
                rectRotated(scr,(255,255,255), [h/2, h/2,r], [30, 3], -i*360/8)
                rectRotated(scr,(255,255,255), [h/2, h/2,r+128], [20, 3], -i*360/8)
                rectRotated(scr,(255,255,255), [h/2, h/2,r+256], [30, 3], -i*360/8)
                rectRotated(scr,(0,255,0), [h/2, h/2, r], [l1, abs(current_db[cols[i]][sample])*2], -i*360/8)

                show_text(scr, str(i+1), font1, h/2-r*3/4*m.sin(-i*360/8/180*m.pi), h/2-r*3/4*m.cos(-i*360/8/180*m.pi), (255,255,255))



            show_text(scr, f'Sample: {sample}', font1, 5/6*h+w/6, 75, color=(255,255,255))            
            show_text(scr, f'Angle: {(angle+1)*5}°', font1, h/2+w/2, 75, color=(255,255,255))
            show_text(scr, 'Pose', font1,h*1/4+w*3/4, h/3,color=(255,255,255))
            rectRotated(scr,(255,255,255), [h*1/4+w*3/4, h/3+75, -25], [60,60],0)
            show_text(scr, f'{current_db[cols[8]][sample]}', font1, h*1/4+w*3/4, h/3+72)
            show_text(scr, 'Repetition', font1,h*1/4+w*3/4, h/2,color=(255,255,255))
            rectRotated(scr,(255,255,255), [h*1/4+w*3/4, h/2+70, -25], [60,60],0)
            show_text(scr, f'{current_db[cols[9]][sample]}', font1, h*1/4+w*3/4, h/2+66)


            events = pygame.event.get()
            for event in events:
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            
            pygame_widgets.update(events)
            pygame.display.update()
    except KeyboardInterrupt:
        exit()

if __name__ == "__main__":

    folder_path = "/media/tullio/SDHD/Tesi Magistrale/Programmi/DATA/RAW/" #ALWAY add end slash

    db_by_angle = load_dataset(folder_path)
    main(db_by_angle)