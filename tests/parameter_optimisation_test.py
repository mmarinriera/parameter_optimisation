import random
import subprocess
import sys

def mutation_function(current_value, amplitude, val_range):
    if(val_range[0] == val_range[1]):
        return current_value
    found = False
    while (not(found)):
        new_value = random.uniform(current_value - 0.5 * amplitude, current_value + 0.5 * amplitude)
        if(new_value > val_range[0] and new_value < val_range[1]):
            found = True
    return new_value

codename = sys.argv[1]
ic_file = "sphere_ic.vtk"
executable = "./sphere_intercalation.e"
target_file = "sphere_raster/sphere_raster_1.0_1.0.Tf_1.vtk"
founder_param_set = [0.0, 0.0]
distance_threshold = 0.5
log_mode = True
model_output = "false"

identity = "founder"

label = codename + "_" + identity

# we determine the distance between founder and target
model_call = [executable, ic_file, target_file, label, "true"] + [str(i) for i in founder_param_set]
result = subprocess.run(model_call, stdout=subprocess.PIPE)

founder_distance_to_target = float(result.stdout.decode('utf-8'))
print("founder distance to target =", founder_distance_to_target)

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
    print("START of generation", generation, "current distance to target", distance_to_target)
    # we mutate param 1 first
    if(log_mode):
        logfile.write("Generation: " + str(generation) + " Individual: " + str(individual) + "\n")

    new_param_set = [mutation_function(param_set[0],0.2,[0,param_set[1]]), param_set[1]]

    identity = "gen" + str(generation) + "_" + "ind" + str(individual)
    label = codename + "_" + identity

    model_call = [executable, ic_file, target_file, label, model_output] + [str(i) for i in new_param_set]
    result = subprocess.run(model_call, stdout=subprocess.PIPE)
    new_distance_to_target = float(result.stdout.decode('utf-8'))

    if(log_mode):
        logfile.write("- Current minimal distance " + str(distance_to_target) + "\n")
        logfile.write("- Mutating first parameter " + str(new_param_set) + "\n")
        logfile.write("\tNew distance to target: " + str(new_distance_to_target) + "\n")
        print("- Current minimal distance ", distance_to_target)
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

    # we do the search for param 2 now
    new_param_set = [param_set[0], mutation_function(param_set[1],0.2,[param_set[0],1.0])]

    identity = "gen" + str(generation) + "_" + "ind" + str(individual)
    label = codename + "_" + identity

    model_call = [executable, ic_file, target_file, label, model_output] + [str(i) for i in new_param_set]
    result = subprocess.run(model_call, stdout=subprocess.PIPE)
    new_distance_to_target = float(result.stdout.decode('utf-8'))

    if(log_mode):
        logfile.write("- Current minimal distance " + str(distance_to_target) + "\n")
        logfile.write("- Mutating second parameter " + str(new_param_set) + "\n")
        logfile.write("\tNew distance to target: " + str(new_distance_to_target) + "\n")
        print("- Current minimal distance ", distance_to_target)
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
