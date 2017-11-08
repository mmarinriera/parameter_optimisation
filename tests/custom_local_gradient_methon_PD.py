import random
import subprocess
import sys
from copy import copy

def mutation_set_value(current_value, amplitude, val_range):
    if(val_range[0] == val_range[1]):
        return current_value

    found = False
    while (not(found)):
        new_value = random.uniform(current_value - 0.5 * amplitude, current_value + 0.5 * amplitude)
        if(new_value > val_range[0] and new_value < val_range[1]):
            found = True
    return new_value

def mutation_function_PD(base_params, param_index):
    value_to_change = base_params[param_index]
    if (param_index == 0):
        new_value = value_to_change #mutation_set_value(value_to_change, 0.2, [0, 1])
    elif (param_index == 1):
        new_value = mutation_set_value(value_to_change, 0.2, [0, base_params[2]])
    elif (param_index == 2):
        new_value = mutation_set_value(value_to_change, 0.2, [base_params[1], 1])
    elif (param_index == 3):
        new_value = mutation_set_value(value_to_change, 0.2, [0, base_params[4]])
    else:
        new_value = mutation_set_value(value_to_change, 0.2, [base_params[3], 1])

    new_params = copy(base_params)
    new_params[param_index] = new_value
    return new_params

codename = sys.argv[1]
ic_file = "sphere_ic.vtk"
executable = "./sphere_intercalation_PD.e"
target_file = "output/target_limb_like_0.75_0.0_1.0_0.0_0.5.Tf_0.vtk"
n_params = 5
founder_param_set = [0.75, 1.0, 1.0, 1.0, 1.0]
distance_threshold = 0.2
log_mode = False
model_output = "true"

identity = "founder"

label = codename + "_" + identity

# we determine the distance between founder and target
model_call = [executable, ic_file, target_file, label, "true"] + [str(i) for i in founder_param_set]
print("model_call", model_call)
result = subprocess.run(model_call, stdout=subprocess.PIPE)

founder_distance_to_target = float(result.stdout.decode('utf-8'))

param_set = founder_param_set
distance_to_target = founder_distance_to_target
generation = 0
individual = 0

if (log_mode):
    print("Log mode : ON")
    log_name = codename + ".log"
    logfile = open(log_name,"w")
    logfile.write("Founder data.\n")
    logfile.write("ic file: " + ic_file + "\n")
    logfile.write("target shape file: " + target_file + "\n")
    logfile.write("distance threshold: " + str(distance_threshold) + "\n")
    logfile.write("founder parameter set: " + str(founder_param_set) + "\n")
    logfile.write("founder distance to target: " + str(founder_distance_to_target) + "\n")
    model_output = "true"

while (distance_to_target > distance_threshold):
    print("START of generation", generation, "current distance to target =", distance_to_target)
    for individual in range(0, n_params):
        # we go through a loop mutating one parameter at a time
        if(log_mode):
            logfile.write("Generation: " + str(generation) + " Individual: " + str(individual) + "\n")

        new_param_set = mutation_function_PD(param_set, individual)

        if(new_param_set[individual] == param_set[individual]):
            if(log_mode):
                logfile.write("- Parameter cannot be mutated, skip individual\n")
                print("- Parameter cannot be mutated, skip individual")
            continue

        identity = "gen" + str(generation) + "_" + "ind" + str(individual)
        label = codename + "_" + identity

        model_call = [executable, ic_file, target_file, label, model_output] + [str(i) for i in new_param_set]
        result = subprocess.run(model_call, stdout=subprocess.PIPE)
        new_distance_to_target = float(result.stdout.decode('utf-8'))

        if(log_mode):
            logfile.write("- Current minimal distance " + str(distance_to_target) + "\n")
            logfile.write("- Mutating " + str(individual) + "th parameter " + str(new_param_set) + "\n")
            logfile.write("\tNew distance to target: " + str(new_distance_to_target) + "\n")
            print("- Current minimal distance ", distance_to_target)
            print("- Mutating", str(individual),"th parameter")
            print("\tNew distance to target: ", new_distance_to_target)

        if(new_distance_to_target < distance_to_target):
            distance_to_target = new_distance_to_target
            param_set = new_param_set
            if(log_mode):
                logfile.write("\t\tSelected.\n")
                print("\t\tSelected")
            if(distance_to_target < distance_threshold):
                opt_param_set = param_set
                model_call = [executable, ic_file, target_file, label + ".optimal", "true"] + [str(i) for i in opt_param_set]
                result = subprocess.run(model_call, stdout=subprocess.PIPE)
                continue

    generation += 1
    if(log_mode):
        logfile.write("\n")
        print("")

if(log_mode):
    logfile.close()

print("CLOSE ENOUGH!")
