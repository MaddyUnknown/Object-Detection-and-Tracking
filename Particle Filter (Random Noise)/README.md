# Particle Filter with random noise


## Object Detection used:
Simple Blob analysis since the video used is a controled simulation and background remains constant along with the obstracle.

## Motion model:
Random noise allowing the particles to move randomly from the nth location to the (n+1)th location.

## Observation:
Not ideal for real-world motion prediction, specially when the object is undetected for a long time.


## Sample Image:

![img](https://github.com/MaddyUnknown/Object-Detection-and-Tracking/blob/main/Particle%20Filter%20(Random%20Noise)/Particle%20view.gif)

## Library used:
- `OpenCV 4.1.1`
- `Numpy 1.19.2`
- `Matplotlib 3.1.1`
