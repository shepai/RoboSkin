# Examples
This folder contains examples for using the library. If you do not have a TacTip/optical tactile sensor built you can use a virtual recorded one.

## flow
Flow demonstrates trying to predict where the points are moving. This provides directional data to the system. 

<img src="https://github.com/shepai/RoboSkin/blob/main/Assets/images/movementVector.gif">

## OpticFLow 
This shows the difference btween he flow algorithm and an optic flow algorithm. Both have heir pros and cons.

## Pressure
This example breaks the sensed area into receptive fields - this is represented be an n $\times$ n grid. 

<img src="https://github.com/shepai/RoboSkin/blob/main/Assets/images/Push.gif">

## Robot leg
Robot leg provides the intrefacing with a Raspberry Pi Pico that is hooked up to a kitronics robot hat. The servos are wired to outputs 1, 2 and 3. The start positions are manually set in the Leg class to match the robot leg we made. You can easily edit this to match start positions for your own in the initilizer class. 

<img width="30%" src="https://github.com/shepai/RoboSkin/blob/main/Assets/images/runAway.gif">

## Minimal simulation
This example makes use of a virtual tactip and attempts to map simulated environmental information onto the tactip. Areas that come into contact with the surface will expand, as if it was getting closer to the camera. 

