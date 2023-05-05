import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt
path= letRun.path
#videoFile=path+"Movement4.avi"
skin=sk.Skin(device=0) #load skin object using demo video
frame=skin.getFrame()
old_T=skin.origin

#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
time=0
while(True):
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
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            #d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            v[i] =np.array([x1-x2,y1-y2])
            cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 0, 0), thickness=2)#
    av=np.sum(v,axis=0)/len(v)
    #cv2.putText(new,"CENTRE",(int(av[0]),int(av[1])),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
    #show user the imagesS
    sk=skin.getFrame()
    cv2.imshow('spots', new)
    cv2.imshow('our binary',im)
    cv2.imshow('unprocessed', sk)
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break
    elif q & 0xFF == ord('r'):
        skin.origin=skin.zero()

    """fig, axis = plt.subplots(1,3, figsize=(3.5, 2.0))

    # For Sine Function
    axis[0].imshow(sk)
    axis[0].set_title("A", loc="left")
    axis[0].xaxis.set_visible(False)
    axis[0].yaxis.set_visible(False)
    # For Cosine Function
    axis[1].imshow(new)
    axis[1].set_title("B", loc="left")
    axis[1].xaxis.set_visible(False)
    axis[1].yaxis.set_visible(False)
    # For Tangent Function
    axis[2].arrow(0, 0, av[1], av[0], head_width=0.2, head_length=0.1, length_includes_head=True, facecolor="black")
    #plt.subplot(2,2,4)
    axis[2].set_title("C", loc="left")
    axis[2].set_xlim((-1.1, 1.1))
    axis[2].set_ylim((-1.1, 1.1))
    axis[2].set_aspect("equal")
    
    # Combine all the operations and display
    fig.tight_layout(pad=0.05)
    fig.savefig("/its/home/drs25/Pictures/DUMP/TACTILE POINT/"+"save"+str(time)+".png")"""
    time+=1
    #out.write(np.concatenate((sk,new),axis=1)) #uncomment to record video
skin.close()
cv2.destroyAllWindows()
#out.release() #uncomment to record video