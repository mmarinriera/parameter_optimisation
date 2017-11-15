import pygmo as pg
import subprocess
import sys
import time

executable = "./sphere_intercalation_PD.e"
target_file = "PD_targets/target_limb_like_0.75_0.0_1.0_0.0_0.5.Tf_0.vtk"
n_params = 5

#Problem definition###########################################################################
class sphere_intercalation:

    # _sphere_intercalation__dim = n_params

    def __init__(self, dim = n_params, own_arg = 0):
        self.own_id = own_arg
        self.dim = dim

    # Reimplement the virtual method that defines the objective function.
    def fitness(self, x):

        model_call = [executable, target_file, "dummy", "false", str(self.own_id)] + [str(i) for i in x]
        result = subprocess.run(model_call, stdout=subprocess.PIPE)

        f = float(result.stdout.decode('utf-8'))

        return [f]

    def get_bounds(self):
        return ([0] * self.dim, [1] * self.dim)

    def get_name(self):
        return "Simulation of sphere_intercalation with 2 different tissue domains"

    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def get_extra_info(self):
        return "\n\t Problem dimension: " + str(self.__dim)

################################################################################################

if __name__ == "__main__":
    code_name = sys.argv[1]
    pop_size = int(sys.argv[2])
    n_islands = int(sys.argv[3])
    n_generations_per_iter = int(sys.argv[4])
    n_iter = int(sys.argv[5])
    stop_threshold = 0.40
    run_identifier = code_name + "-p" + str(pop_size) + "-i" + str(n_islands)\
    + "-g" + str(n_generations_per_iter * n_iter)

    print("code_name:", code_name)
    print("pop. size =", pop_size, "n. islands =", n_islands,\
    "gens. per iter. =", n_generations_per_iter, "n. iter. =", n_iter)

    log_name = run_identifier + ".log"
    logfile = open(log_name,"w")
    logfile.write("General data.\n")
    logfile.write("target shape file: " + target_file + "\n")
    logfile.write("distance threshold: " + str(stop_threshold) + "\n")

    prob = pg.problem(sphere_intercalation())
    # prob = pg.problem(pg.schwefel(5))

    algo = pg.algorithm(pg.pso(gen = n_generations_per_iter, variant = 5))

    start_time = time.time()

    archi = pg.archipelago(n = n_islands, algo = algo, prob = prob, pop_size = pop_size)

    print("time for initialisation", time.time() - start_time,"s")

    for i in range(n_iter):
        start_time = time.time()

        archi.evolve(n_generations_per_iter)
        archi.wait()

        print("iteration", i, "took", time.time() - start_time,"s")

        logfile.write("After " + str(n_generations_per_iter * (i + 1)) + " iterations\n")
        optimum_reached = min([isl.get_population().champion_f[0] for isl in archi]) < stop_threshold

        if(optimum_reached):
            logfile.write("\tOPTIMUM REACHED!" + "\n")

        count = 0

        for isl in archi:
            # print("Island ", count)
            # print("\tFinal champion distance =", isl.get_population().champion_f[0])
            # print("\tFinal champion params =\n\t", isl.get_population().champion_x)
            # print("\n")

            logfile.write("\tIsland " + str(count) + "\n")
            logfile.write("\tChampion distance = " + str(isl.get_population().champion_f[0]) + "\n")
            logfile.write("\tChampion params =\n\t" + str(isl.get_population().champion_x) + "\n")
            logfile.write("\n")

            if(i == n_iter - 1 or optimum_reached):
                code_name = run_identifier + "_" + "island-" + str(count) + "_champion"
                model_call = [executable, target_file, code_name, "true", "0"] + [str(i) for i in isl.get_population().champion_x]
                # print("champion call", model_call)
                result = subprocess.run(model_call, stdout=subprocess.PIPE)

            count += 1

        if(optimum_reached):
            break
    logfile.close()
