import main

def show(genomes, config):
    nets = []
    gen = []
    rockets = []

    for _, g in genomes:
        net = main.neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        rockets.append(main.Rocket(100, 200))
        g.fitness = 0
        gen.append(g)

    bg = main.BGMove(0)
    asteroids = [main.Asteroid(800)]
    win = main.pygame.display.set_mode((main.WIDTH, main.HEIGHT))
    score = 0
    clock = main.pygame.time.Clock()

    run = True
    while run:
        clock.tick(60)
        for event in main.pygame.event.get():
            if event.type == main.pygame.QUIT:
                run = False
                main.pygame.quit()
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
            t_softmax_result = main.t_softmax(net_output)
            class_output = main.np.argmax(((t_softmax_result / main.torch.max(t_softmax_result)) == 1))

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
            asteroids.append(main.Asteroid(800))

        for r in remove:
            asteroids.remove(r)
        for x, rocket in enumerate(rockets):
            if rocket.y + rocket.img.get_height() >= main.HEIGHT or rocket.y < 0:
                rockets.pop(x)
                nets.pop(x)
                gen.pop(x)

        bg.move()
        main.draw_window(win, rockets, asteroids, bg, score)

def runSaveLoad(config_path):

    config = main.neat.config.Config(main.neat.DefaultGenome, main.neat.DefaultReproduction,
                                main.neat.DefaultSpeciesSet, main.neat.DefaultStagnation, config_path)
    pop = main.neat.Population(config)
    pop.add_reporter(main.neat.StdOutReporter(True))
    stats = main.neat.StatisticsReporter()
    pop.add_reporter(stats)
    winner = pop.run(main.fitness, 100)
    path = main.os.path.join("assets", "winner_.pkl")
    with open(path, "wb") as f:
        main.pickle.dump(winner, f)
        f.close()

    # Load the configuration again
    config = main.neat.config.Config(main.neat.DefaultGenome, main.neat.DefaultReproduction, main.neat.DefaultSpeciesSet,
                                main.neat.DefaultStagnation, config_path)

    # Open the pickle file again
    with open(path, "rb") as f:
        genome = main.pickle.load(f)

    # Create a list with the first item being the loaded genome
    genomes = [(1, genome)]

    # With this genome, create the NN again
    show(genomes, config)


if __name__ == "__SaveLoadDisplay__":
    local_dir = main.os.path.dirname(__file__)
    config_path = main.os.path.join(local_dir, "config.txt")
    runSaveLoad(config_path)