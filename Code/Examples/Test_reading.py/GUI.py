import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from control import Board,Experiment
import time
import RoboSkin as sk
import cv2
import numpy as np
B=Board()

#get serial boards and connect to first one
COM=""
while COM=="":
    try:
        res=B.serial_ports()
        print("ports:",res)
        B.connect(res[0])
        B.runFile("C:/Users/dexte/github/RoboSkin/Code/Examples/Test_reading.py/mOTRO CONTROL.py")
        COM=res[0]
    except IndexError:
        time.sleep(1)
exp=Experiment(B,device=1)

pygame.init()
(width, height) = (1000, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('TacTip frame control software')
running = True
background_colour = (255,255,255)
screen.fill(background_colour)
pygame.display.flip()

#Create slider 1
slider = Slider(screen, 100, 100, 300, 10, min=0, max=99, step=1)
output = TextBox(screen, 450, 100, 50, 50, fontSize=30)

slider_width = 20
slider_height = 300
slider_x = width // 2 - slider_width // 2
slider_y = height // 2 - slider_height // 2
slider_range = height - slider_height

# Create slider 2
sliderZ = Slider(screen, 100, 200, 300, 10, min=0, max=99, step=1)
outputZ = TextBox(screen, 450, 200, 50, 50, fontSize=30)

output.disable()  # Act as label instead of textbox

def speed():
    global mode
    global outer_i
    global outer_j
    
    outer_i=0
    outer_j=0
    mode="speed"

def pressure():
    global mode
    global outer_i
    global outer_j
    
    outer_i=0
    outer_j=0
    mode="pressure"

def direction():
    global running
    running = False

def sho_():
    global image
    #display normal
    frame=exp.skin.getFrame()
    img=frame.copy()
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    screen.blit(surf, (550, 0))
    #display direction
    im=exp.skin.getBinary()
    new=np.ones_like(frame) * 255
    t_=exp.skin.getDots(im)
    t=exp.skin.movement(t_)
    v=np.zeros(t.shape)
    if t.shape[0]>2:
        new[t[:,0],t[:,1]]=(0,255,0)
        new[old_T[:,0],old_T[:,1]]=(0,0,255)
        for i,coord in enumerate(t): #show vectors of every point
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            v[i] =np.array([x1-x2,y1-y2])
            cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 0, 0), thickness=2)#

    resized = cv2.resize(new, dim, interpolation = cv2.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    screen.blit(surf, (550+height+50, 0))
    #display force
    image=exp.skin.getForce(im,SPLIT,image=image,threshold=20,degrade=20,) #get the force push
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    resized = cv2.resize(tactile, dim, interpolation = cv2.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    screen.blit(surf, (550, width+50))
    #update
    if mode=="menu": 
        pygame_widgets.update(events)
    pygame.display.update()

#create buttons for experiments
b1 = Button(
    screen, 100, 400, 140, 25, text='Speed experiment',
            fontSize=20, margin=20,
            inactiveColour=(255, 0, 0),
            pressedColour=(0, 255, 0), radius=20,
            onClick=lambda: speed()
            
         )

b2 = Button(
    screen, 100, 430, 140, 25, text='Pressure experiment',
            fontSize=20, margin=20,
            inactiveColour=(255, 0, 0),
            pressedColour=(0, 255, 0), radius=20,
            onClick=lambda: pressure()
            
         )

b3 = Button(
    screen, 100, 460, 140, 25, text='Direction experiment',
            fontSize=20, margin=20,
            inactiveColour=(255, 0, 0),
            pressedColour=(0, 255, 0), radius=20,
            onClick=lambda: direction()
            
         )
currentVal=50
currentValZ=50

#initial variables for skin reading
past_Frame=exp.skin.getBinary()
old_T=exp.skin.origin
image=np.zeros_like(past_Frame)
SPLIT=10

outer_i=0
outer_j=0
mode="menu"
samples=2
trials=1
speeds=[10,20,30]
a=[]
Image=None

frame=exp.skin.getFrame()
img=frame.copy()
scale_percent=40
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)


while running:
    events=pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
    if mode=="menu":
        val=slider.getValue()
        if val!=currentVal: #change
            mov=val-currentVal
            print("Move by:",mov)
            currentVal=val
            B.moveX(mov)
        val=sliderZ.getValue()
        if val!=currentValZ: #change
            mov=val-currentValZ
            print("Move by:",mov)
            currentValZ=val
            B.moveZ(mov)
        output.setText(slider.getValue())
        outputZ.setText(sliderZ.getValue())

    screen.fill(background_colour)

    #display tactip imagery
    sho_()

    if mode=="pressure":
        #pressure experiment
        print(outer_j)
        if outer_j==0: #first trial
            Image=exp.skin.getBinary() #get initial image
            exp.move_till_touch(Image) #be touching the platform
            sho_()
        if outer_j==samples: #simulated for loop
            outer_j=0
            outer_i+=1
        im=exp.skin.getBinary()
        exp.image=exp.skin.getForce(im,exp.SPLIT,image=exp.image,threshold=20,degrade=20,past=Image) #get the force push
        sho_()
        time.sleep(1)
        exp.moveZ(outer_i,-1) #move down
        sho_()
        time.sleep(1)
        im=exp.skin.getBinary()
        exp.image=exp.skin.getForce(im,exp.SPLIT,image=exp.image,threshold=20,degrade=20,past=Image) #get the force push
        a.append(np.sum(exp.image)/(exp.SPLIT*exp.SPLIT*255))
        exp.moveZ(outer_i,1) #move back
        sho_()
        outer_j+=1
        if outer_i==trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(a)
            a=[]
            exp.moveZ(2,1) #move back
    elif mode=="speed":
        #speed experiment
        if outer_j==0:
            exp.skin.reset()
            #self.moveX(1,-1)
            old_T=exp.skin.origin
            sho_()
        if outer_j==len(speeds): #simulated for loop
            outer_j=0
            outer_i+=1
            exp.b.setSpeed(20)
        sp=speeds[outer_j]
        exp.b.setSpeed(sp)
        sho_()
        exp.moveZ(0.5,-1) #move down
        sho_()
        exp.moveX(0.5-(outer_j/10),1)
        sho_()
        vectors=exp.read_vectors(old_T)
        #r_dt[j]=np.sum(np.linalg.norm(vectors))
        time.sleep(1)
        exp.moveZ(0.5,1) #move up
        sho_()
        exp.moveX(0.5-(outer_j/10),-1)
        sho_()
        time.sleep(1)
        sho_()

        outer_j+=1
        if outer_i==trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(a)
            a=[]
            exp.b.setSpeed(20)
    elif mode=="edges":
        #edges experiment
        if outer_j==samples: #simulated for loop
            outer_j=0
            outer_i+=1
            
        outer_j+=1
        if outer_i==trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(a)
            a=[]

    if mode=="menu": 
        pygame_widgets.update(events)
    pygame.display.update()