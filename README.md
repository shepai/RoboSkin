# RoboSkin
This is a library for interfacing with optical tactile skin sensors for robotics projects.

Sensing an environment is seen within many adaptive agents, and much of this within biology is tactile. High resolution tactile sensing allows environmental sensing over larger surface areas as well as more detailed information about the surfaces. This library has the support for reading interpreting tactile information.

## Constructing the sensor

### Tactip
The <a href="https://softroboticstoolkit.com/tactip">TacTip</a>, developed from BRL, parts were 3D printed and consisted of a camera mount, main body, tip and ring. The camera mount was edited on CAD software to mount our webcam (a cheap standard Logitech USB webcam), the main body would attach on to the mound via a series of screws.

The camera was mounted onto a custom 3D printed part that connected to a repurposed USB wide angle lens webcam. The webcam had the circular vision in the centre, which would need cropping at a later stage. The power of the LED ring was connected to the power of the USB camera.  

<img src="Assets/images/mounting the ring.png" width="50%">

The skins produced through silicone within a <a href="https://github.com/shepai/RoboSkin/blob/main/Assets/3D%20files/Skin/MouldTIp.stl">3D printed plastic mould</a>. This mould was measured to fit the silicone inside a thin layer, with small holes in to create the tips that would later be painted white.

<img src="Assets/images/RENDER.png" width="50%">


Silicone was made and dyed black to prevent light interference. After being poured into the mould and left for twenty-four hours, a solid yet flexible tactip was produced. The The mould required a lubricant spray over to prevent the tips getting pulled off. To paint the tips we used a thin layer of plastic sheet with an acrylic paint. The tip was turned inside out (so the tips were on the exterioir) and gently dabbed over the paint.  

After drying we poured a clear silicone gel into the tactip and left to set in a vacuum chamber for an hour, and then left in an oven at 80 degrees centigrade for 2.5 hours to cure. This gave the tip some force against a surface so it would not cave in under force.  

### General skin


### parts list
You will need access to 3D printing, an oven and laser cutting. The softrobotics toolkit has a better part list for the construction of the TacTip. 

The mould for the tactip required an industrial 3D printer for accuracies for 1mm diameter holes. 

The silicone used:
Skin - 1:1 mix
Gel - RTV27905 silicone gel

Paint - high viscosity acrylic white paint off Amazon

Lens - lasercut plastic

Superglue 

Silicone lubricant spray

Camera - wide angle lens usb camera from amazon


## Using the library

To import the library simply

```
import RoboSkin as sk
skin=sk.Skin()
```

If you wish to use a virtual tactip check out our dataset within <a>Assets/Video demos</a>. This can be imported via a parameter

```
import RoboSkin as sk
skin=sk.Skin(videoFile=path+"Movement4.avi")
```

We can gather frame using ```skin.getFrame()``` to return what the camera of video is viewing. The video will play on repeat till you ```skin.release()```. You can also get a processed image using ```skin.getBinary()``` which returns a noise reduced binary image. 

### Vector prediction

The library estimates the movement of points between two frames. It takes an origin frame when you initialize the object. This attempts to maximize the amount of points it is reading. This function requires a binary image that has had been converted to an array of centroid points (n,2). We use the ```skin.getDots(image)``` function to gather this. It may not return the same size as your ```skin.origin``` points. The ```skin.movement(points)``` function will turn a centroid array of (n,2) and try and map each of these points to an origin array of (m,2) so each index between origin and the new point array represents that specific point. 

```
im=skin.getBinary() #get image
t_=skin.getDots(im) #get the entroid points in the image
t=skin.movement(t_) #get the prediccted points 
```

The distance between these points is calculated by the following equation (Euclidean distance) where o is the origin points and t is the mapped points. It calculates the distances of points o and t. We pass matrices through the equations to calculate all distances.

$d(o,t) = \sqrt{\sum_{i=1}^{n}(o_i-t_i)^2}$ 

The result of plotting this is below:

<img src="Assets/images/movementVector.gif">

It is not a perfect method however helps find the general jist of movement.

### Push prediction

We may want to predict where the push is coming from within the sensor. This is done by chosing a grid size that will represent how large the receptive fields are. If our grid size is 5 then the tactile image will be viewed as a 5 $\times$ 5 image. 

The force within each grid point (denoted in a matrix that startes empty and recursevly enters this function $P$) calculates the difference between the frames $F$ at each index within the selected grid square Where the bounds are set by dimentions x and y ($d_x,d_y$). This is then averaged, with the global average subtracted from this to highlight change. Finally we subtract $\gamma$ which represents a temporal dampner that reduces pixels over time. This is how we get the faded affect as a stimuli drops. 

$P_{j:d_x,i:d_y} = \frac{\sum_{j}^{d_x}\sum_{i}^{d_y} \left | F_{t,i,j} - F_{t+1,i,j}\right | }{(d_x-j)*(d_y-i)} - \gamma -\frac{\left | F_{t} - F_{t+1}\right |}{n}$

```
past_Frame=skin.getBinary()
image=np.zeros_like(past_frame)
SPLIT=25 #25x25 receptive fields
time.sleep(0.2) #time between
im=skin.getBinary()
image=skin.getForce(im,past_Frame,SPLIT,image=image) #get the force push
past_Frame=im.copy()
```


<img src="Assets/images/Push.gif">
