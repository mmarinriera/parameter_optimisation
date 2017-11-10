import subprocess

code_name = "benchmark_pso_var5"
pop = [5, 10, 20, 40]
n_islands = 5
n_gen_per_iter = 10
n_iter = 10
n_replicates = 5

executable = "parameter_optimisation_pygmo2_parallel.py"

for p in pop:
    for i in range(n_replicates):
        call = ["python", executable, code_name + ".replicate" + str(i), \
        str(p), str(n_islands), str(n_gen_per_iter), str(n_iter)]

        run = subprocess.Popen(call)
        run.wait()
