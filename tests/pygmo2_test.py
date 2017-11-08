#Adapted from :
#https://esa.github.io/pagmo2/docs/python/tutorials/coding_udp_simple.html

import pygmo as pg

class sphere_function:
    def fitness(self, x):
        return [sum(x*x)]

    def get_bounds(self):
        return ([-1,-1],[1,1])

prob = pg.problem(sphere_function())
algo = pg.algorithm(pg.sade(gen=100))
if __name__ == "__main__":
    archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=20)
    archi.evolve(10)
    archi.wait()
    res = [isl.get_population().champion_f for isl in archi]
    print(res)
