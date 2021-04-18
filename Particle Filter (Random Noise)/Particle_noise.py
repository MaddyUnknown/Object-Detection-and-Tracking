###################################
#           Imports               #
###################################
import cv2
import numpy as np
import time
import math
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

###################################
#       Utility Functions         #
###################################

def create_ground_truth(img):
    '''
    Create the background image by setting the objects location pixals as white
    '''
    img = np.array(img)
    img[0:200, 0: 200 , :] = np.ones((200, 200, 3))*255
    return img

def diff_mask(image, background):
    ''' Subtract the current image (with the object) with the background image to get a blob mask '''
    diff = cv2.subtract(background, image)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    return res

def find_region(mask):
    ''' Use the blob mask to find the bouding box of the image '''
    rows = np.where(np.any(mask!=0, axis=1))[0]
    columns = np.where(np.any(mask!=0, axis=0))[0]
    try:
        top_left = (min(columns), min(rows))
        bottom_right = (max(columns), max(rows))
    except:
        top_left = (-1,-1)
        bottom_right = (-1, -1)
    return top_left, bottom_right

def find_center(bound_box):
    ''' Find center of the detected image '''
    if not((bound_box[0][0] == -1 and bound_box[0][1] == -1) or (bound_box[1][0] == -1 and bound_box[1][1] == -1) ):
        x = int((bound_box[1][0]+bound_box[0][0])/2)
        y = int((bound_box[1][1]+bound_box[0][1])/2)
        return (x,y)
    else:
        return (-1,-1)

###################################
#         Particle Class          #
###################################

class Particle:
    __world_size = 800
    def __init__(self, bound=None):
        if bound:
            self.x = random.randint(bound[0][0], bound[1][0])
            self.y = random.randint(bound[0][1], bound[1][1])
        else:
            self.x = int(random.random()*Particle.__world_size)
            self.y = int(random.random()*Particle.__world_size)
        self.forward_noise = 10          # variance of the noise
        self.measurement_noise = 4.0    # variance of the mesurement (needed to calcualte the gaussian)
    def set_xy(self, x, y):
        ''' Set the coordinate of x and y for the particle '''
        self.x = x
        self.y = y
    def check_bound(self , x,y):
        ''' Check postion to find of the particle is outside the world boundaries '''
        if x>=Particle.__world_size:
            x = Particle.__world_size - 1
        elif x<=0:
            x = 1
        if y>=Particle.__world_size:
            y = Particle.__world_size - 1
        elif y<=0:
            y = 1
        return x,y
    def apply_noise(self):
        '''Apply motion noise for randomness '''
        noise = (int(random.gauss(0.0,self.forward_noise)))
        return noise
    def motion_model(self):
        ''' Function defining the motion model for predicting the next possible position '''
        x = int(self.x +  self.apply_noise())
        y = int(self.y +  self.apply_noise())
        x,y = self.check_bound(x,y)
        r = Particle()
        r.set_xy(x,y)
        return r
    def Gaussian(self, mu, var, x):
        ''' Gaussian function to get a normalised weight for accuracy '''
        return math.exp(- ((mu - x) ** 2) / var*2.0) / math.sqrt(2.0 * math.pi * var)
    def measurement_prob(self, center):
        ''' Probabilty of whether the particle is near the actual object (coordinate specified by the center) '''
        dist = math.sqrt((center[0]-self.x)**2 + (center[1]-self.y)**2)
        prob = self.Gaussian(0.0, self.measurement_noise, dist)
        return prob
    def __str__(self):
        return "[x: %4d y: %4d ]"%(self.x, self.y)
    def set_world_size(num):
        ''' Change the world_size parameter '''
        Particle.__world_size = num

###################################
#          Filter Class           #
###################################

class Particle_Filter:
    def __init__(self, bound=None, no_particles=1024):
        self.num_particle = no_particles
        self.p = []
        for i in range(self.num_particle):
            self.p.append(Particle(bound))    
    def weight(self, center):
        ''' Calculate the weight for all the Particles in filter '''
        w = []
        for i in range(self.num_particle):
            w.append(self.p[i].measurement_prob(center))
        return w
    def forward(self):
        ''' Shift the Particle by the specified model or external input '''
        new = []
        for i in range(self.num_particle):
            new.append(self.p[i].motion_model())
        self.p = new
    def resampling(self, w):
        ''' Resample the particles according to weight (w) '''
        new_sample = []
        index = [i for i in range(self.num_particle)]
        new_index = random.choices(index, weights=w, k=self.num_particle)
        for i in new_index:
            new_sample.append(self.p[i])
        self.p = new_sample
    def print_particle(self, num=10):
        ''' Print the first n particles '''
        print("First %4d particles"%(num))
        for i in range(num):
            print(self.p[i])
    def __getitem__(self, index):
        ''' Get the contents of ith particle '''
        return self.p[index]
    def particle_plot(self, frame):
        ''' Plot the particles in a given frame '''
        for i in range(self.num_particle):
            cv2.circle(frame, (self.p[i].x, self.p[i].y), 1, (255,0,0), 2)
        return frame 
    def plot3d(self, center, weight):
        ''' 3d plot of the given (x,y) points with weight along the z axis '''
        x = []
        y = []
        for i in range(self.num_particle):
            x.append(self.p[i].x)
            y.append(self.p[i].y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, weight)
        print(center[0], center[1])
        ax.scatter3D(center[0],center[1],0, color='red')
        #ax.set_zlim(0.0, 0.1)
        plt.show()

###################################
#         Driver Function         #
###################################

def main():
    cap = cv2.VideoCapture('Simulation.mp4')        # Import the simulation video

    if(cap.isOpened() == False):
        print("Error opening video")

    ret, frame = cap.read()
    background = create_ground_truth(frame)         # Create the background

    ############ Init #############
    particles = Particle_Filter()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            diff = diff_mask(frame,background)
            frame_out = np.array(frame)
            bound_box = find_region(diff)
            center = find_center(bound_box)
            with_rect = cv2.rectangle(frame, bound_box[0], bound_box[1], (255, 0, 0), 2)
            
            ###########  Particle Filter Apply ###################
            t = time.time()
            particles.forward()
            particles.particle_plot(frame_out)
            if(center != (-1, -1)):
                w = particles.weight(center)
                particles.resampling(w)
            #print(center)
            #particles.print_particle(2)
            print("Time required to process frame: %.3f seconds"%(time.time()-t))
            ######################################################

            cv2.imshow("particle view", frame_out)
            cv2.imshow("video", cv2.resize(with_rect, (200, 200)))
            cv2.imshow("detector", cv2.resize(diff,(200,200)))
            
            if cv2.waitKey(20) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__  == '__main__':
    main()