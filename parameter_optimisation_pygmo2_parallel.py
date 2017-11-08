import pygmo as pg
import subprocess

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
        return "Simulation of sphere_intercalation with 2 different tissu domains"

    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def get_extra_info(self):
        return "\n\t Problem dimension: " + str(self.__dim)

################################################################################################

pop_size = 5
n_generations_per_iter = 1
n_iter = 1
n_islands = 1
stop_threshold = 0.40
run_identifier = "codename" + "-p" + str(pop_size) + "-g" + str(n_generations_per_iter * n_iter)


log_name = run_identifier + ".log"
logfile = open(log_name,"w")
logfile.write("General data.\n")
logfile.write("target shape file: " + target_file + "\n")
logfile.write("distance threshold: " + str(stop_threshold) + "\n")

prob = pg.problem(sphere_intercalation())
# prob = pg.problem(pg.schwefel(5))

algo = pg.algorithm(pg.pso(gen = n_generations_per_iter, variant = 5))

if __name__ == "__main__":
    archi = pg.archipelago(n=2,algo=algo, prob=prob, pop_size=pop_size)
    # emigration = migration.best_s_policy(0.2,migration.rate_type.fractional) #emigrant selection policy
    # replacement = migration.fair_r_policy(0.2,migration.rate_type.fractional) #replacement of locals by immigrants

    for i in range(n_iter):
        archi.evolve(n_generations_per_iter)
        archi.wait()

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