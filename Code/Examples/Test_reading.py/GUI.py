import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from control import Board
import time

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


pygame.init()
(width, height) = (640, 480)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('TacTip frame control software')
running = True
background_colour = (255,255,255)
screen.fill(background_colour)
pygame.display.flip()

slider = Slider(screen, 100, 100, 300, 10, min=0, max=99, step=1)
output = TextBox(screen, 450, 100, 50, 50, fontSize=30)

slider_width = 20
slider_height = 300
slider_x = width // 2 - slider_width // 2
slider_y = height // 2 - slider_height // 2
slider_range = height - slider_height

# Create the slider
sliderZ = Slider(screen, 100, 200, 300, 10, min=0, max=99, step=1)
outputZ = TextBox(screen, 450, 200, 50, 50, fontSize=30)

output.disable()  # Act as label instead of textbox

currentVal=50
currentValZ=50
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
    pygame_widgets.update(events)
    pygame.display.update()