from PyGMO.problem import base
from PyGMO import *
import subprocess

executable = "./sphere_intercalation_PD.e"
target_file = "PD_targets/target_limb_like_0.75_0.0_1.0_0.0_0.5.Tf_0.vtk"
n_params = 5

#Problem definition###########################################################################
class sphere_intercalation(base):
    """
    Basic sphere of tissue with cell intercalation that parametrises the cell-
    level orientations for the intercalation process
    """

    _sphere_intercalation__dim = n_params

    def __init__(self, dim = n_params, own_arg = 0):
        # First we call the constructor of the base class telling PyGMO
        # what kind of problem to expect ('dim' dimensions, 1 objective, 0 contraints etc.)
        super(sphere_intercalation,self).__init__(dim)

        # We set the problem bounds (in this case equal for all components)
        self.set_bounds(0, 1)
        self.own_id = own_arg

    # Reimplement the virtual method that defines the objective function.
    def _objfun_impl(self, x):

        model_call = [executable, target_file, "dummy", "false", str(self.own_id)] + [str(i) for i in x]
        result = subprocess.run(model_call, stdout=subprocess.PIPE)

        f = float(result.stdout.decode('utf-8'))

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (f, )


    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

################################################################################################

pop_size = 10
n_generations_per_iter = 10
n_iter = 5
n_islands = 10
stop_threshold = 0.40
run_identifier = "pso_var5_migration-0.2_topo-ring_benchmark" + "-p" + str(pop_size) + "-g" + str(n_generations_per_iter * n_iter)



log_name = run_identifier + ".log"
logfile = open(log_name,"w")
logfile.write("General data.\n")
logfile.write("target shape file: " + target_file + "\n")
logfile.write("distance threshold: " + str(stop_threshold) + "\n")


# Define the problem
prob = sphere_intercalation(dim = n_params)
# prob = problem.schwefel(10)
algo = algorithm.pso(gen = n_generations_per_iter, variant = 5)

archi = archipelago(topology=topology.ring())
emigration = migration.best_s_policy(0.2,migration.rate_type.fractional) #emigrant selection policy
replacement = migration.fair_r_policy(0.2,migration.rate_type.fractional) #replacement of locals by immigrants

for i in range(n_islands):
    archi.push_back(island(algo, prob, pop_size, s_policy = emigration, r_policy = replacement))
    # archi.push_back(island(algorithm.jde(gen = n_generations_per_iter, ftol = 0.31), prob, pop_size))

for i in range(1, n_iter):
    archi.evolve(n_generations_per_iter)  # Evolve the island
    archi.join()

    count = 0
    logfile.write("After " + str(n_generations_per_iter * i) + " iterations\n")
    for isl in archi:
        # print("Island ", count)
        # print("\tFinal champion distance =", isl.population.champion.f)
        # print("\tFinal champion params =\n\t", isl.population.champion.x)
        # print("\n")

        logfile.write("\tIsland " + str(count) + "\n")
        logfile.write("\tChampion distance = " + str(isl.population.champion.f) + "\n")
        logfile.write("\tChampion params =\n\t" + str(isl.population.champion.x) + "\n")
        logfile.write("\n")

        if(i == 5):
            code_name = run_identifier + "_" + "island-" + str(count) + "_champion"
            model_call = [executable, target_file, code_name, "true", "0"] + [str(i) for i in isl.population.champion.x]
            # print("champion call", model_call)
            result = subprocess.run(model_call, stdout=subprocess.PIPE)

        count += 1

logfile.close()
