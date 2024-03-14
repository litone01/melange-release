from scripts.solver import HeteroAcceleratorSolver

def solver_run_example():
    #### Input required for the solver ####
    #### Replace the following with your own input ####
    # based on the offline profiling results
    gpu_info_example = {
        "A10G": {
            "cost": 1.01,
            "tputs": [[5, 1], [10, 5]],
        },
        "A100": {
            "cost": 3.67,
            "tputs": [[20, 2], [50, 20]],
        },
    }
    # based on the analysis of the request distribution
    workload_distribution = [[0.25, 0.5], [0.25, 0.25]]
    # parameters
    overall_rate = 16
    slice_factor = 1

    #### Run the solver ####
    mix_result = HeteroAcceleratorSolver(
        workload_distribution=workload_distribution,
        overall_rate=overall_rate,
        slice_factor=slice_factor,
        gpu_info=gpu_info_example,
    ).run()
    print(mix_result)


if __name__ == "__main__":
    solver_run_example()
