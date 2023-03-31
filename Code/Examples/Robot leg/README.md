# Robot leg example
This example uses a 3 degrees of freedom robotic leg/arm with a TacTip mounted on the end.  
The leg was made up of hobby servos each having a draw of approximately 500 milliamps, and 100 milliamps at rest. The main arm was controlled by a Raspberry Pi Pico receiving serial commands through a USB.  

Initial experimentation would move the knee and head joint in proportion, and eave the hip rotation out, essentially making the leg 2 DOF. This allowed experimentation on a 1D axis where the TacTip could move back and forth across the surface, given an amount to move by $d$. The rotational degrees were calculated as $\theta$.  

$\theta = arcsin(\frac{d}{d_{initial}})$ 

The knee would be calculated by adding $\theta$ and the head joint would subtract $\theta$. Though an inverse kinematics approach would be more accurate this method was done for simplicity.  



## bounce
The leg will move up and down the axis, encountering the surface at the furthest extension of the leg. If the tactip was intercepted by an object it would detect this, and if the object had a higher force than the motors, the leg would move upwards. This effectively demonstrated the TacTip as an analogue push sensor. 

<img src="./././Assets/images/armTouch.gif">

## Move away

The reading of pressure is broken up into quarters and each corner summed. The highest value would determine what direction the leg would move in. Using pre-determined logic, the arm would move away from the stimuli. This made use of the hip joint.  

<img src="https://github.com/shepai/RoboSkin/blob/main/Assets/images/runAway.gif">
