#include <curand_kernel.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "../bolls/include/dtypes.cuh"
#include "../bolls/include/inits.cuh"
#include "../bolls/include/links.cuh"
#include "../bolls/include/meix.cuh"
#include "../bolls/include/polarity.cuh"
#include "../bolls/include/property.cuh"
#include "../bolls/include/solvers.cuh"
#include "../bolls/include/vtk.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1f;
const auto n_max = 150000;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;
float mean_proliferation_rate = 0.015f;
// float min_proliferation_rate = std::stof(argv[2]);
int n_time_steps = 400;

enum Cell_types { mesenchyme, epithelium };

__device__ Cell_types* d_type;
__device__ int* d_NC;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
// __device__ float* d_prolif_rate;

MAKE_PT(Cell, w, theta, phi);

__device__ Cell force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        if (Xi.w < 0.f) Xi.w = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 2.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.1f;

    if (d_type[j] == epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    if (Xi.w<0.f) Xi.w=0.f;
    if (d_type[i] == mesenchyme or d_type[j] == mesenchyme) return dF;

    dF += rigidity_force(Xi, r, dist) * 0.15;
    return dF;
}

__global__ void proliferate(float mean_rate, float mean_distance, Cell* d_X,
    int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * mean_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - mean_rate))
        return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case mesenchyme: {
            return;
        }
        case epithelium: {
            if (d_epi_nbs[i] > 14) return;
            if (d_mes_nbs[i] < 1) return;
            auto rnd = curand_uniform(&d_state[i]);
            if (rnd > mean_rate) return;

        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w / 2;
        d_X[i].w = d_X[i].w / 2;
    } else {
        d_X[n].w = d_X[i].w;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_NC[n] = d_NC[i];
}

__global__ void update_protrusions(const int n_cells, float* d_params_x,
    float* d_params_y, const Grid<n_max>* __restrict__ d_grid,
    const Cell* __restrict d_X, curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rand_nb_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
    auto b =
        d_grid->d_point_id[d_grid->d_cube_start[rand_nb_cube] +
                           min(static_cast<int>(
                                   curand_uniform(&d_state[i]) * cells_in_cube),
                               cells_in_cube - 1)];
    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];

    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto not_initialized = link->a == link->b;
    auto position = d_NC[a];
    float x_ratio = d_params_x[position];
    float y_ratio = d_params_y[position];
    int x_y_or_z;
    if(curand_uniform(&d_state[i]) < x_ratio)
        x_y_or_z = 0;
    else if(curand_uniform(&d_state[i]) < y_ratio)
        x_y_or_z = 1;
    else
        x_y_or_z = 2;

    auto more_along_x = false;
    auto more_along_y = false;
    auto more_along_z = false;
    more_along_x = fabs(new_r.x / new_dist) > fabs(old_r.x / old_dist);
    more_along_y = fabs(new_r.y / new_dist) > fabs(old_r.y / old_dist);
    more_along_z = fabs(new_r.z / new_dist) > fabs(old_r.z / old_dist);

    if (not_initialized or (x_y_or_z == 0 and more_along_x)
        or (x_y_or_z == 1 and more_along_y) or
        (x_y_or_z == 2 and more_along_z)) {
    // if(not_initialized){
        link->a = a;
        link->b = b;
    }
}

template<typename Pt, int n_max, template<typename, int> class Solver,
    typename Prop>
void fill_solver_w_epithelium(Solution<Pt, n_max, Solver>& inbolls,
    Solution<Pt, n_max, Solver>& outbolls, Prop& type, unsigned int n_0 = 0)
{
    assert(n_0 < *inbolls.h_n);
    assert(n_0 < *outbolls.h_n);

    int j = 0;
    for (int i = 0; i < *inbolls.h_n; i++) {
        if (type.h_prop[i] == epithelium) {
            outbolls.h_X[j].x = inbolls.h_X[i].x;
            outbolls.h_X[j].y = inbolls.h_X[i].y;
            outbolls.h_X[j].z = inbolls.h_X[i].z;
            outbolls.h_X[j].phi = inbolls.h_X[i].phi;
            outbolls.h_X[j].theta = inbolls.h_X[i].theta;
            j++;
        }
    }
    *outbolls.h_n = j;
}

//*****************************************************************************

int main(int argc, char const* argv[])
{
    // std::clock_t time_0 = clock();

    std::string ic_file_name = "sphere_ic.vtk";
    std::string opt_file_name = argv[1];
    std::string codename = argv[2];
    std::string output_mode = argv[3];
    bool write_output = false;
    if(output_mode == "true")
        write_output = true;
    std::string device_idx = argv[4];
    cudaSetDevice(std::stoi(device_idx));
    std::string arg1 = argv[5];
    // std::string param2 = argv[6];
    // std::string param3 = argv[7];
    // std::string param4 = argv[8];
    // std::string param5 = argv[9];
    // std::string output_tag = codename + "_" + param1 + "_" + param2 + "_"
    //     + param3 + "_" + param4 + "_" + param5;
    std::string output_tag = codename + "_" + arg1;

    auto n_partitions = std::stoi(arg1);
    auto three_d = false;
    auto n_compartments = 0;
    if (three_d)
        n_compartments = pow(n_partitions, 3);
    else
        n_compartments = pow(n_partitions, 2);

    std::cout<<"n_compartments = " << n_compartments << std::endl;


    float raw_params[n_compartments][2];
    for (auto i = 0; i < n_compartments; i++){
        raw_params[i][0] = rand()/float(RAND_MAX);
        raw_params[i][1] = rand()/float(RAND_MAX);
    }

    // raw_params[0][0] = 0.0f;
    // raw_params[0][1] = 0.5f;
    // raw_params[1][0] = 0.0f;
    // raw_params[1][1] = 0.5f;
    // raw_params[2][0] = 0.0f;
    // raw_params[2][1] = 0.5f;
    // raw_params[3][0] = 0.0f;
    // raw_params[3][1] = 0.5f;
    // raw_params[4][0] = 0.0f;
    // raw_params[4][1] = 0.5f;
    // raw_params[5][0] = 0.0f;
    // raw_params[5][1] = 0.5f;
    // raw_params[6][0] = 0.0f;
    // raw_params[6][1] = 0.5f;
    // raw_params[7][0] = 0.0f;
    // raw_params[7][1] = 0.5f;
    // raw_params[8][0] = 0.0f;
    // raw_params[8][1] = 0.5f;
    // raw_params[9][0] = 0.0f;
    // raw_params[9][1] = 0.5f;
    // raw_params[10][0] = 0.0f;
    // raw_params[10][1] = 0.5f;
    // raw_params[11][0] = 0.0f;
    // raw_params[11][1] = 0.5f;
    // raw_params[12][0] = 0.0f;
    // raw_params[12][1] = 1.0f;
    // raw_params[13][0] = 0.0f;
    // raw_params[13][1] = 1.0f;
    // raw_params[14][0] = 0.0f;
    // raw_params[14][1] = 1.0f;
    // raw_params[15][0] = 0.0f;
    // raw_params[15][1] = 1.0f;


    float *params_x = (float*)malloc(n_compartments * sizeof(float));
    float *params_y = (float*)malloc(n_compartments * sizeof(float));

    for (auto i = 0; i < n_compartments; i++){
        params_x[i] = raw_params[i][0];
        params_y[i] = raw_params[i][0] + raw_params[i][1]
            * (1 - raw_params[i][0]);
        std::cout<<"params "<< params_x[i] << " " << params_y[i]<<std::endl;
    }
    float *d_params_x;
    float *d_params_y;
    cudaMalloc((float**)&d_params_x, n_compartments * sizeof(float));
    cudaMalloc((float**)&d_params_y, n_compartments * sizeof(float));

    cudaMemcpy(d_params_x, params_x, n_compartments * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_params_y, params_y, n_compartments * sizeof(float),
        cudaMemcpyHostToDevice);


    // Load the initial conditions
    Vtk_input input(ic_file_name);
    int n0 = input.n_bolls;
    Solution<Cell, n_max, Grid_solver> sphere(n0);

    input.read_positions(sphere);
    input.read_polarity(sphere);

    // std::clock_t time_1 = clock();

    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    Property<n_max, int> intype;

    input.read_property(intype, "cell_type");  // we read it as an int, then we translate to
                                               // enum "Cell_types"
    for (int i = 0 ; i < n0 ; i++) {
        sphere.h_X[i].w = 0.0f;
        if (intype.h_prop[i] == 0) {
            type.h_prop[i] = mesenchyme;
        } else if (intype.h_prop[i] == 1) {
            type.h_prop[i] = epithelium;
            if (sphere.h_X[i].x > 2.5f)
                sphere.h_X[i].w = 1.0f;
        }
    }

    sphere.copy_to_device();
    type.copy_to_device();

    Property<n_max, int> NC("NC");
    cudaMemcpyToSymbol(d_NC, &NC.d_prop, sizeof(d_NC));

    float3 max{-1000.f};
    float3 min{1000.f};
    for (auto i = 1; i < n0; i++) {
        max.x = std::max(max.x, sphere.h_X[i].x);
        max.y = std::max(max.y, sphere.h_X[i].y);
        max.z = std::max(max.z, sphere.h_X[i].z);
        min.x = std::min(min.x, sphere.h_X[i].x);
        min.y = std::min(min.y, sphere.h_X[i].y);
        min.z = std::min(min.z, sphere.h_X[i].z);
    }
    float3 length = max - min;
    float3 compartment_length = length * (1 / float(n_partitions));

    // std::cout<<"max "<<max.x<<" "<<max.y<<" "<< max.z<<std::endl;
    // std::cout<<"min "<<min.x<<" "<<min.y<<" "<< min.z<<std::endl;
    // std::cout<<"length "<<length.x<<" "<<length.y<<" "<< length.z<<std::endl;
    // std::cout<<"C length "<<compartment_length.x<<" "<<compartment_length.y<<" "<< compartment_length.z<<std::endl;

    for (auto i = 0; i < n0; i++) {
        // x coordinate
        auto idx_x = n_partitions - 1;
        for (auto k = 0; k < n_partitions - 1; k++) {
            if(sphere.h_X[i].x >= min.x + compartment_length.x * k and
                sphere.h_X[i].x <= min.x + compartment_length.x * (k + 1)) {
                idx_x = k;
                break;
            }
        }
        // y coordinate
        auto idx_y = n_partitions - 1;
        for (auto k = 0; k < n_partitions - 1; k++) {
            if(sphere.h_X[i].y >= min.y + compartment_length.y * k and
                sphere.h_X[i].y <= min.y + compartment_length.y * (k + 1)) {
                idx_y = k;
                break;
            }
        }
        if(three_d){
            // z coordinate
            auto idx_z = n_partitions - 1;
            for (auto k = 0; k < n_partitions - 1; k++) {
                if(sphere.h_X[i].z >= min.z + compartment_length.z * k and
                    sphere.h_X[i].z <= min.z + compartment_length.z * (k + 1)) {
                    idx_z = k;
                    break;
                }
            }
            // auto n_compartments = pow(n_partitions, 3);
            NC.h_prop[i] = pow(n_partitions, 2) * idx_x + n_partitions * idx_y
                + idx_z;
        } else {
            // auto n_compartments = pow(n_partitions, 2);
            NC.h_prop[i] = n_partitions * idx_x + idx_y;
        }
    }
    NC.copy_to_device();

    // std::cout << "initial nbolls " << n0 << " nmax " << n_max << std::endl;

    Property<n_max, int> n_mes_nbs("n_mes_nbs");  // defining these here so function
    Property<n_max, int> n_epi_nbs("n_epi_nbs");  // "neighbour_init" can see them
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // Declaration of links
    Links<static_cast<int>(n_max * prots_per_cell)> protrusions(
        protrusion_strength, n0 * prots_per_cell);
    auto intercalation =
    std::bind(link_forces<static_cast<int>(n_max * prots_per_cell), Cell>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    Grid<n_max> grid;

    // State for links
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(
        n_max, seed, d_state);


    // std::clock_t time_2 = clock();
    Vtk_output output(output_tag, false);
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        sphere.copy_to_host();
        protrusions.copy_to_host();
        type.copy_to_host();
        NC.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop,
            n_mes_nbs.d_prop + sphere.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + sphere.get_d_n(), 0);

        protrusions.set_d_n(sphere.get_d_n() * prots_per_cell);
        grid.build(sphere, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            sphere.get_d_n(), d_params_x, d_params_y, grid.d_grid, sphere.d_X,
            protrusions.d_state, protrusions.d_link);

        sphere.take_step<force>(dt, intercalation);

        proliferate<<<(sphere.get_d_n() + 128 - 1) / 128, 128>>>(
            mean_proliferation_rate, r_min, sphere.d_X,
            sphere.d_n, d_state);

        if(write_output) {
            output.write_positions(sphere);
            output.write_links(protrusions);
            output.write_polarity(sphere);
            output.write_property(type);
            output.write_property(NC);
        }
    }
    // std::clock_t time_3 = clock();

    sphere.copy_to_host();
    type.copy_to_host();

    Solution<Cell, n_max, Grid_solver> epi_Tf(n0);
    fill_solver_w_epithelium(sphere, epi_Tf, type);
    epi_Tf.copy_to_device();
    Vtk_output output_epi_T0(output_tag + ".Tf", false);
    if(write_output) {
        Vtk_output output_epi_T0(output_tag + ".Tf", false);
        output_epi_T0.write_positions(epi_Tf);
    }

    // std::clock_t time_4 = clock();

    // std::cout<<"LOADING OPTIMAL SHAPE"<<std::endl;
    //reading the optimal shape and making the shape comparison
    Vtk_input opt_input(opt_file_name);
    Solution<Cell, n_max, Grid_solver> optimum(opt_input.n_bolls);
    opt_input.read_positions(optimum);
    optimum.copy_to_device();

    // std::clock_t time_5 = clock();

    Meix shape_comparison;
    float distance = shape_comparison.shape_comparison_distance_bolls_to_bolls(
        optimum, epi_Tf);
    std::cout << distance << std::endl;

    // std::clock_t time_6 = clock();
    // std::cout<<"time elapsed"<<std::endl;
    // std::cout<<"\tcheckp-1: "<<float(time_1-time_0)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"\tcheckp-2: "<<float(time_2-time_0)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"\tcheckp-3: "<<float(time_3-time_0)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"\tcheckp-4: "<<float(time_4-time_0)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"\tcheckp-5: "<<float(time_5-time_0)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"\tcheckp-6: "<<float(time_6-time_0)/CLOCKS_PER_SEC<<std::endl;

    return 0;
}
