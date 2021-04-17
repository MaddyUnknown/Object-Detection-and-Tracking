import pygame
import time
class Sim():
    def __init__(self):
        self.bound_x = 800
        self.bound_y = 800
        win = pygame.display.set_mode((self.bound_x, self.bound_y))
        pygame.display.set_caption("Simulation")
        self.x=50
        self.y=50
        run = True
        time.sleep(2)
        while run:
            pygame.time.delay(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    pygame.image.save(win, "screenshot.jpeg")
            self.x, self.y = self.physics(self.x, self.y)
            win.fill((255, 255, 255))
            pygame.draw.circle(win, (255, 0, 0), (self.x,self.y),20)
            pygame.draw.rect(win, (125, 125, 125), (400, 400 , 100, 150))
            pygame.display.update()

        pygame.quit()

    def bound(func):
        def m(self, x, y):
            x, y = func(self, x, y)
            if(x>(self.bound_x-20)):
                x = self.bound_x-20
            elif(x<=0):
                x = 0

            if(y>(self.bound_y-20)):
                y = self.bound_y-20
            elif(y<=0):
                y = 0
            return x, y
        return m

    @bound
    def physics(self, x, y, vel=1):
        x += vel
        y += vel
        return x, y
            

Sim()