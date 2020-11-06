from neat.math_util import softmax
import pickle
import pygame
import os
import random
import neat
import SaveLoadDisplay
import numpy as np
import torch
from torch import softmax

pygame.font.init()

WIDTH = 800
HEIGHT = 600

dirname = os.path.dirname(__file__)
ROCKET_IMGS = [pygame.transform.scale(pygame.image.load(os.path.join("assets", "rocket1.png")), (112, 112)),
               pygame.transform.scale(pygame.image.load(os.path.join("assets", "rocket2.png")), (112, 112)),
               pygame.transform.scale(pygame.image.load(os.path.join("assets", "rocket3.png")), (112, 112))]
ASTEROID_IMGS = pygame.transform.scale(pygame.image.load(os.path.join("assets", "asteroid.png")), (100, 100))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "background.png")))
STAT_FONT = pygame.font.SysFont("dejavuserif", 30)

class Rocket:
    IMGS = ROCKET_IMGS
    ANIMATION_TIME = 10

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def flyup(self):
        self.velocity = -10
        self.tick_count = 0

    def flydown(self):
        self.velocity = 10
        self.tick_count = 0

    def stay(self):
        self.velocity = 0
        self.tick_count = 0

    def move(self):
        self.tick_count += 1
        displacement = self.velocity

        self.y = self.y + displacement

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        new_rect = self.img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(self.img, new_rect)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Asteroid:
    VELOCITY = 20

    def __init__(self, xcoord):
        self.xcoord = xcoord
        self.obj = pygame.transform.flip(ASTEROID_IMGS, False, True)
        self.top = 0
        self.bot = 0
        self.passed = False
        self.set_ycoord()
        self.set_pos()
        self.pos = 0

    def set_ycoord(self):
        self.ycoord = random.randrange(10, 550)
        self.top = self.ycoord - self.obj.get_height()
        self.bot = self.xcoord - self.obj.get_height()

    def move(self):
        self.xcoord -= self.VELOCITY

    def set_pos(self):
        self.pos = (self.xcoord, self.ycoord)

    def draw(self, win):
        win.blit(self.obj, (self.xcoord, self.ycoord))

    def collide(self, rocket):
        rocket_mask = rocket.get_mask()
        top_mask = pygame.mask.from_surface(self.obj)
        top_offset = (self.xcoord - rocket.x, self.ycoord - round(rocket.y))
        t_point = rocket_mask.overlap(top_mask, top_offset)

        if t_point:
            return True
        else:
            return False


class BGMove:
    VELOCITY = 1
    WIDTH = BG_IMG.get_width()
    IMG = BG_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw_bg(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, rockets, asteroids, bg, score):
    bg.draw_bg(win)
    text = STAT_FONT.render("Skóre: " + str(score), True, (255, 255, 255))
    win.blit(text, (WIDTH - 10 - text.get_width(), 10))
    for asteroid in asteroids:
        asteroid.draw(win)
    for rocket in rockets:
        rocket.draw(win)
    pygame.display.update()


def fitness(genomes, config):
    nets = []
    gen = []
    rockets = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        rockets.append(Rocket(100, 250))
        g.fitness = 0
        gen.append(g)

    bg = BGMove(0)
    asteroids = [Asteroid(800)]
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    score = 0
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        asteroid_index = 0
        if len(rockets) > 0:
            if len(asteroids) > 1 and rockets[0].x > asteroids[0].xcoord + asteroids[0].obj.get_width():
                asteroid_index = 1
        else:
            run = False
            break

        for x, rocket in enumerate(rockets):
            gen[x].fitness += 0.1
            rocket.move()

            net_output = nets[rockets.index(rocket)].activate((rocket.y, abs(rocket.y - asteroids[asteroid_index].ycoord), abs(rocket.y - asteroids[asteroid_index].xcoord)))
            t_softmax_result = t_softmax(net_output)
            class_output = np.argmax(((t_softmax_result / torch.max(t_softmax_result)) == 1))

            if class_output == 0:
                rocket.flyup()
            if class_output == 1:
                rocket.flydown()
            if class_output == 2:
                rocket.stay()

        add_asteroid = False
        remove = []

        for asteroid in asteroids:
            for x, rocket in enumerate(rockets):
                if asteroid.collide(rocket):
                    gen[x].fitness -= 5
                    rockets.pop(x)
                    nets.pop(x)
                    gen.pop(x)

                if not asteroid.passed and asteroid.xcoord < rocket.x:
                    asteroid.passed = True
                    add_asteroid = True

            if asteroid.xcoord + asteroid.obj.get_width() < 0:
                remove.append(asteroid)

            asteroid.move()

        if add_asteroid:
            score += 1
            print(score)
            for g in gen:
                g.fitness += 0.5
            asteroids.append(Asteroid(800))

        for r in remove:
            asteroids.remove(r)
        for x, rocket in enumerate(rockets):
            if rocket.y + rocket.img.get_height() >= HEIGHT or rocket.y < 0:
                rockets.pop(x)
                nets.pop(x)
                gen.pop(x)

        bg.move()
        draw_window(win, rockets, asteroids, bg, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)


    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    winner = pop.run(fitness, 200)
    path = os.path.join("assets", "winner.pkl")
    with open(path, "wb") as f:
        pickle.dump(winner, f)
        f.close()
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    with open(path, "rb") as f:
        genome = pickle.load(f)
    genomes = [(1, genome)]
    SaveLoadDisplay.show(genomes, config)

def t_softmax(x):
    x_in = torch.FloatTensor(x)
    m = softmax(x_in, 0)
    return m


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
