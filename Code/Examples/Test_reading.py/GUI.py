import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from control import Board
import time
import RoboSkin as sk
import cv2
import numpy as np
B=Board()
skin=sk.Skin(device=1)
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
    global running
    running = False
    import speed_exp

def pressure():
    global running
    running = False
    import pressure_exp

def direction():
    global running
    running = False
    import edges_exp

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
past_Frame=skin.getBinary()
old_T=skin.origin
image=np.zeros_like(past_Frame)
SPLIT=10

while running:
    events=pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
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
    frame=skin.getFrame()
    img=frame.copy()
    scale_percent=40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    
    screen.blit(surf, (550, 0))
    #display direction
    im=skin.getBinary()
    new=np.ones_like(frame) * 255
    t_=skin.getDots(im)
    t=skin.movement(t_)
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

    #dislpay force
    image=skin.getForce(im,SPLIT,image=image,threshold=20,degrade=20,) #get the force push
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    resized = cv2.resize(tactile, dim, interpolation = cv2.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    screen.blit(surf, (550, width+50))

    pygame_widgets.update(events)
    pygame.display.update()