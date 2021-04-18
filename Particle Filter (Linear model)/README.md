# Particle Filter with random noise


## Object Detection used:
Simple Blob analysis since the video used is a controled simulation and background remains constant along with the obstracle.

## Motion model:
Linear motion (along with tiny random motions to handle small zig-zag motion done by realworld objects) allowing the particles to move randomly from the nth location to the (n+1)th location.

## Observation:
Better than Random noise model for modeling real-world object showing overall linear motion.

## Sample Image:

![img](https://github.com/MaddyUnknown/Object-Detection-and-Tracking/blob/main/Particle%20Filter%20(Linear%20model)/Particle_view.gif)

## Library used:
- `OpenCV 4.1.1`
- `Numpy 1.19.2`
- `Matplotlib 3.1.1`
