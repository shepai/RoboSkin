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
import matplotlib.pyplot as plt

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
screen=None
slider=None
sliderZ=None
output=None
outputZ=None
(width1, height2) = (1000, 600)
slider_width = 20
slider_height = 300
slider_x = width1 // 2 - slider_width // 2
slider_y = height2 // 2 - slider_height // 2
slider_range = height2 - slider_height
running = True
background_colour = (255,255,255)
b1=None
b1_save=None
b1_sho=None
b2=None
b2_save=None
b2_sho=None
b3=None
b3_save=None
b3_sho=None

def init():
    global screen
    global slider, output
    global sliderZ, outputZ
    global b1, b2, b3, b1_save, b2_save, b3_save, b1_sho, b2_sho, b3_sho

    pygame.init()
    
    screen = pygame.display.set_mode((width1, height2))
    pygame.display.set_caption('TacTip frame control software')
    
    screen.fill(background_colour)
    pygame.display.flip()
    #Create slider 1
    slider = Slider(screen, 100, 100, 300, 10, min=0, max=99, step=1)
    output = TextBox(screen, 450, 100, 50, 50, fontSize=30)

    # Create slider 2
    sliderZ = Slider(screen, 100, 200, 300, 10, min=0, max=99, step=1)
    outputZ = TextBox(screen, 450, 200, 50, 50, fontSize=30)

    output.disable()  # Act as label instead of textbox
    #create buttons for experiments
    b1 = Button(
        screen, 100, 400, 140, 25, text='Speed experiment',
                fontSize=20, margin=20,
                inactiveColour=(130, 50, 209),
                pressedColour=(0, 255, 0), radius=20,
                onClick=lambda: change_mode("speed")
                
            )

    b1_save = Button(
        screen, 260, 400, 80, 25, text='Save',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: save("speed")
                
            )

    b1_sho = Button(
        screen, 360, 400, 80, 25, text='Plot',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: plot("speed")
                
            )

    b2 = Button(
        screen, 100, 430, 140, 25, text='Pressure experiment',
                fontSize=20, margin=20,
                inactiveColour=(130, 50, 209),
                pressedColour=(0, 255, 0), radius=20,
                onClick=lambda: change_mode("pressure")
                
            )

    b2_save = Button(
        screen, 260, 430, 80, 25, text='Save',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: save("pressure")
                
            )

    b2_sho = Button(
        screen, 360, 430, 80, 25, text='Plot',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: plot("pressure")
                
            )

    b3 = Button(
        screen, 100, 460, 140, 25, text='Direction experiment',
                fontSize=20, margin=20,
                inactiveColour=(130, 50, 209),
                pressedColour=(0, 255, 0), radius=20,
                onClick=lambda: change_mode("edges")
                
            )

    b3_save = Button(
        screen, 260, 460, 80, 25, text='Save',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: save("edges")
                
            )

    b3_sho = Button(
        screen, 360, 460, 80, 25, text='Plot',
                fontSize=20, margin=20,
                inactiveColour=(255, 0, 0),
                pressedColour=(255, 255, 0), radius=20,
                onClick=lambda: plot("edges")
                
            )

init()

def change_mode(mo):
    global mode
    global outer_i
    global outer_j
    
    outer_i=0
    outer_j=0
    mode=mo



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

def save(id): #save the data if it has beenritten to
    if id=="speed" and np.sum(data_speed)!=0:
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/speed",data_speed)
    if id=="pressure" and np.sum(data_pressure)!=0:
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/pressure",data_pressure)
    if id=="edges" and np.sum(data_edge)!=0:
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/edges",data_edge)

def plot(id): #plot the different datas
    if id=="speed" and np.sum(data_speed)!=0:
        for i in range(len(data_speed)):
            plt.plot(speeds,data_speed[i],label="Exp"+str(i))
        plt.xlabel("Speed setting")
        plt.ylabel("Average magnitude size (px)")
        plt.title("How speed affects the time of arrival to the magnitude of vectors")
        plt.legend(loc="lower left")
    if id=="pressure" and np.sum(data_pressure)!=0:
        for i in range(len(data_pressure)):
            plt.plot([i for i in np.arange(0, CM, ST)],data_pressure[i])
        plt.xlabel("Pressure in cm")
        plt.ylabel("Pressure in summed pixels")
        plt.title("Pressure vs cm")
    if id=="edges" and np.sum(data_edge)!=0:
        for i in range(len(data_edge)):
            f=data_edge[i][0]
            l=data_edge[i][1]
            r=data_edge[i][2]
            weights = np.arange(1, len(f)+1)
            plt.scatter(f[:,0],f[:,1],c=weights, cmap='Purples')
            plt.scatter(l[:,0],l[:,1],c=weights, cmap='Greens')
            plt.scatter(r[:,0],r[:,1],c=weights, cmap='Reds')
        plt.xlabel("Speed setting")
        plt.ylabel("Average magnitude size (px)")
        plt.title("How speed affects the time of arrival to the magnitude of vectors")
        plt.legend(loc="lower left")
    plt.show()
    init()
    
init()

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
samples=1
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
CM=1.5
ST=0.5
#data storage
data_speed=np.zeros((samples,len(speeds)))
data_edge=np.zeros((samples,3,samples,2))
data_pressure=np.zeros((samples,len(np.arange(0, CM, ST))))

while running:
    #add colouring if data exists
    if np.sum(data_speed)!=0: 
        b1_save.inactiveColour=(0,255,0)
        b1_sho.inactiveColour=(0,255,0)
    if np.sum(data_pressure)!=0: 
        b2_save.inactiveColour=(0,255,0)
        b2_sho.inactiveColour=(0,255,0)
    if np.sum(data_edge)!=0: 
        b3_save.inactiveColour=(0,255,0)
        b3_sho.inactiveColour=(0,255,0)

    events=pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
    if mode=="menu":
        val=slider.getValue()
        if val!=currentVal: #change
            mov=val-currentVal
            #print("Move by:",mov)
            currentVal=val
            B.moveX(mov)
        val=sliderZ.getValue()
        if val!=currentValZ: #change
            mov=val-currentValZ
            #print("Move by:",mov)
            currentValZ=val
            B.moveZ(mov)
        output.setText(slider.getValue())
        outputZ.setText(sliderZ.getValue())

    screen.fill(background_colour)

    #display tactip imagery
    sho_()

    if mode=="pressure":
        #pressure experiment
        if outer_j==0: #first trial
            sho_()
            Image=exp.skin.getBinary() #get initial image
            exp.move_till_touch(Image) #be touching the platform
            sho_()
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
        data_pressure[outer_i][int(outer_j/ST)]=np.sum(exp.image)/(exp.SPLIT*exp.SPLIT*255)
        outer_j+=ST
        if int(outer_j/ST)>=len(np.arange(0, CM, ST)): #simulated for loop
            outer_j=0
            outer_i+=1
        if outer_i>=trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(data_pressure)
            exp.moveZ(2,1) #move back
    elif mode=="speed":
        #speed experiment
        if outer_j==0:
            sho_()
            exp.skin.reset()
            #self.moveX(1,-1)
            old_T=exp.skin.origin
            sho_()
        sp=speeds[outer_j]
        exp.b.setSpeed(sp)
        sho_()
        exp.moveZ(0.5,-1) #move down
        sho_()
        exp.moveX(0.5-(outer_j/10),1)
        sho_()
        vectors=exp.read_vectors(old_T)
        #r_dt[j]=np.sum(np.linalg.norm(vectors))
        data_speed[outer_i][outer_j]=np.sum(np.linalg.norm(vectors))
        time.sleep(1)
        exp.moveZ(0.5,1) #move up
        sho_()
        exp.moveX(0.5-(outer_j/10),-1)
        sho_()
        time.sleep(1)
        sho_()

        outer_j+=1
        if outer_j==len(speeds): #simulated for loop
            outer_j=0
            outer_i+=1
            exp.b.setSpeed(20)
        if outer_i==trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(data_speed)
            exp.b.setSpeed(20)
    elif mode=="edges":
        #edges experiment
        if outer_j==0:
            sho_()
            exp.skin.reset()
            old_T=exp.skin.origin
            #self.move_till_touch(Image) #be touching the platform
        exp.moveZ(0.5,-1) #move down
        sho_()
        vectors=exp.read_vectors(old_T)
        #fl_dt[i]=np.sum(vectors,axis=0)/len(vectors)
        data_edge[outer_i][0][outer_j]=np.sum(vectors,axis=0)/len(vectors)
        time.sleep(1)
        exp.moveZ(0.5,1) #move up
        sho_()
        time.sleep(1)
        exp.moveZ(0.5,-1) #move down
        sho_()
        exp.moveX(1,-1)
        sho_()
        vectors=exp.read_vectors(old_T)
        #l_dt[i]=np.sum(vectors,axis=0)/len(vectors)
        data_edge[outer_i][1][outer_j]=np.sum(vectors,axis=0)/len(vectors)
        time.sleep(1)
        exp.moveZ(0.5,1) #move up
        sho_()
        exp.moveX(1,1)
        sho_()
        time.sleep(1)
        exp.moveZ(0.5,-1) #move down
        sho_()
        exp.moveX(1,1)
        sho_()
        vectors=exp.read_vectors(old_T)
        #r_dt[i]=np.sum(vectors,axis=0)/len(vectors)
        data_edge[outer_i][2][outer_j]=np.sum(vectors,axis=0)/len(vectors)
        time.sleep(1)
        exp.moveZ(0.5,1) #move up
        sho_()
        exp.moveX(1,-1)
        sho_()
        time.sleep(1)
        outer_j+=1
        if outer_j==samples: #simulated for loop
            outer_j=0
            outer_i+=1
        if outer_i==trials: #finished loop
            mode="menu"
            outer_i=0
            outer_j=0
            print(data_edge)
            

    if mode=="menu": 
        pygame_widgets.update(events)
    pygame.display.update()
    pygame.display.set_mode((width1, height2))

