from PyGMO.problem import base
from PyGMO import algorithm, island, problem
import subprocess

executable = "./sphere_intercalation_PD.e"
target_file = "PD_targets/target_limb_like_0.75_0.0_1.0_0.0_0.5.Tf_0.vtk"
run_identifier = "benchmark-1"
n_params = 5

#functions######################################################################################
def centroid(population):
    centroid = [0.0 for i in range(n_params)]
    for ind in population:
        for i in range(n_params):
            centroid[i] += ind.cur_x[i]
        #print("centroid", centroid)
        #print("individual", ind.cur_x)
    for i in range(n_params):
        centroid[i] *= 1/population_size

    return centroid

def centroid_size(population, centroid):

    csize = 0.0
    for ind in population:
        dist = 0
        for i in range(n_params):
            dist +=(centroid[i]-ind.cur_x[i])**2
        #print("distance", dist)
        csize += dist**0.5
    csize *= 1 / population_size

    return csize
##############################################################################################

#Problem definition###########################################################################
class sphere_intercalation(base):
    """
    Basic sphere of tissue with cell intercalation that parametrises the cell-
    level orientations for the intercalation process
    """

    _sphere_intercalation__dim = n_params

    def __init__(self, dim = n_params):
        # First we call the constructor of the base class telling PyGMO
        # what kind of problem to expect ('dim' dimensions, 1 objective, 0 contraints etc.)
        super(sphere_intercalation,self).__init__(dim)

        # We set the problem bounds (in this case equal for all components)
        self.set_bounds(0, 1)

    # Reimplement the virtual method that defines the objective function.
    def _objfun_impl(self, x):

        model_call = [executable, target_file, "dummy", "false"] + [str(i) for i in x]
        result = subprocess.run(model_call, stdout=subprocess.PIPE)

        f = float(result.stdout.decode('utf-8'))

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (f, )


    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

################################################################################################

population_size = 20
n_generations_per_iter = 1
n_iter = 50
stop_threshold = 0.40



log_name = run_identifier + ".log"
logfile = open(log_name,"w")
logfile.write("General data.\n")
logfile.write("target shape file: " + target_file + "\n")
logfile.write("distance threshold: " + str(stop_threshold) + "\n")


# Define the problem
prob = sphere_intercalation(dim = n_params)
# prob = problem.schwefel(10)

# build a list of algorithms that will be tested on the problem
algo_list = []
algo_list.append(algorithm.de(gen = n_generations_per_iter))
algo_list.append(algorithm.sga(gen = n_generations_per_iter))
algo_list.append(algorithm.sga_gray(gen = n_generations_per_iter))
algo_list.append(algorithm.jde(gen = n_generations_per_iter))
algo_list.append(algorithm.pso(gen = n_generations_per_iter))
algo_list.append(algorithm.bee_colony(gen = n_generations_per_iter))

isl_list = []
for algo in algo_list:
    algorithm_name = algo.__doc__
    print("Testing algorithm:", algorithm_name)
    logfile.write("Testing algorithm: " + algorithm_name)
    isl = island(algo, prob, population_size)  # Instantiate population
    # print("\tSTART OF ITERATIONS")
    f_progression = []
    centroid_size_progression = []
    for i in range(n_iter):
        isl.evolve(1)  # Evolve the island
        isl.join()
        # print("\tGeneration=",i)
        # print("\t\tchampion distance=",island.population.champion.f)
        # print("\t\tchampion params=\n\t\t",island.population.champion.x)
        # print("\t\tcentroid size", centroid_size(island.population, centroid(island.population)))
        # print("\n")
        f_progression.append(isl.population.champion.f)
        centroid_size_progression.append(centroid_size(isl.population, centroid(isl.population)))
        final_iteration = i
        if(isl.population.champion.f[0] <= stop_threshold):
            break

    # print("FINISH!")
    print("\tFinal iteration =", final_iteration)
    print("\tFinal champion distance =", isl.population.champion.f)
    print("\tFinal champion params =\n\t", isl.population.champion.x)
    print("\tDistance progression =\n\t", f_progression)
    print("\tCentroid size progression =\n\t", centroid_size_progression)
    print("\n")

    logfile.write("\tFinal iteration = " + str(final_iteration) + "\n")
    logfile.write("\tFinal champion distance = " + str(isl.population.champion.f) + "\n")
    logfile.write("\tFinal champion params =\n\t" + str(isl.population.champion.x) + "\n")
    logfile.write("\tDistance progression =\n\t" + str(f_progression) + "\n")
    logfile.write("\tCentroid size progression =\n\t" + str(centroid_size_progression) + "\n")
    logfile.write("\n")

    code_name = run_identifier + "_" + algorithm_name.replace(" ","_") + "champion"
    model_call = [executable, target_file, code_name, "true"] + [str(i) for i in isl.population.champion.x]
    # print("champion call", model_call)
    result = subprocess.run(model_call, stdout=subprocess.PIPE)

logfile.close()
