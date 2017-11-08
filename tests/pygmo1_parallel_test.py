from PyGMO.problem import base
from PyGMO import *
import subprocess
import random
from time import sleep

executable = "./sphere_intercalation_PD.e"
target_file = "PD_targets/target_limb_like_0.75_0.0_1.0_0.0_0.5.Tf_0.vtk"
run_identifier = "lolll"

class my_problem(base):
    """
    Basic sphere of tissue with cell intercalation that parametrises the cell-
    level orientations for the intercalation process
    """

    _my_problem__dim = 1

    def __init__(self, dim = 1, own_arg = 0):
        # First we call the constructor of the base class telling PyGMO
        # what kind of problem to expect ('dim' dimensions, 1 objective, 0 contraints etc.)
        super(my_problem,self).__init__(dim)

        # We set the problem bounds (in this case equal for all components)
        self.set_bounds(0, 1)
        self.own_id = own_arg

    # Reimplement the virtual method that defines the objective function.
    def _objfun_impl(self, x):

        #call = ["echo", "'lol'", str(x)]
        #result = subprocess.run(call, stdout=subprocess.PIPE)
        print("own_id",self.own_id)
        sleep(random.uniform(1,2))

        #f = float(result.stdout.decode('utf-8'))

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (random.uniform(0,1), )

class sphere_intercalation(base):
    """
    Basic sphere of tissue with cell intercalation that parametrises the cell-
    level orientations for the intercalation process
    """

    _sphere_intercalation__dim = 5

    def __init__(self, dim = 5, own_arg = 0):
        # First we call the constructor of the base class telling PyGMO
        # what kind of problem to expect ('dim' dimensions, 1 objective, 0 contraints etc.)
        super(sphere_intercalation,self).__init__(dim)

        # We set the problem bounds (in this case equal for all components)
        self.set_bounds(0, 1)

        self.own_id = own_arg

    # Reimplement the virtual method that defines the objective function.
    def _objfun_impl(self, x):

        # print("start", self.own_id)
        model_call = [executable, target_file, "dummy", "false"] + [str(i) for i in x]
        result = subprocess.run(model_call, stdout=subprocess.PIPE)

        f = float(result.stdout.decode('utf-8'))
        # print("end", self.own_id)

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (f, )


    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

population_size = 6
n_generations_per_iter = 1
n_iter = 50

# prob = my_problem(dim = 1, own_arg = "lol")

algo = algorithm.de(gen = n_generations_per_iter)

archi = archipelago(topology=topology.unconnected())
for i in range (6):
    print("island", i)
    archi.push_back(island(algo, sphere_intercalation(dim = 5, own_arg = i), 6))


import time
starting_time = time.time()
archi.evolve(1)
archi.join()
print("elapsed time", time.time() - starting_time)

# for i in range(1,5):
#     print("start Generation",i)
#     # sleep(5)
#     archi.evolve(1)
#     archi.join()
#     # sleep(5)
#     print("end Generation",i)
