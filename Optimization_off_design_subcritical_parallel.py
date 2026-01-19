import time
import json
import multiprocessing as mp
from scipy.optimize import differential_evolution
from Off_design_subcritical import ORC


# Define the objective function at the module level for picklability
def single_objective(x, params_dict):
    T_prod = params_dict['T_prod']
    p_surf = params_dict['p_surf']
    m_gf = params_dict['m_gf']
    T_am = params_dict['T_am']
    T_reinj = params_dict['T_reinj']
    V_d = params_dict['V_d']
    A_eva_d = params_dict['A_eva_d']
    m_wf_d = params_dict['m_wf_d']
    A_cond_d = params_dict['A_cond_d']
    orc_class = params_dict['orc_class']

    p_pump, deltaT_ap_cond = x

    # Create ORC instance with the current parameters
    orc = orc_class(T_prod=T_prod,
                    p_surf=p_surf,
                    m_gf=m_gf,
                    T_ambient=T_am,
                    T_reinj=T_reinj,
                    V_d=V_d,
                    A_eva_d=A_eva_d,
                    m_wf_d=m_wf_d,
                    A_cond_d=A_cond_d,
                    p_pump=p_pump,
                    deltaT_ap_cond=deltaT_ap_cond)

    # Calculate ORC performance
    m_wf, W_elec, eta_ORC, exer_ORC, eta_t_off, T3, T_out_cool, m_cf, symb_eva, symb_cond, pinch_value = orc.ORC_results_geothermal()

    # Check constraint:
    constraint1 = pinch_value
    constraint2 = orc.convergence_threshold - symb_eva
    constraint3 = orc.convergence_threshold - symb_cond
    constraint4 = orc.T_prod - orc.T3 - 0.1

    # Apply penalty if constraint is violated
    penalty1 = 0
    if constraint1 < 0:
        penalty1 = 1e6 * abs(constraint1)

    penalty2 = 0
    if constraint2 < 0:
        penalty2 = 1e6 * abs(constraint2)

    penalty3 = 0
    if constraint3 < 0:
        penalty3 = 1e6 * abs(constraint3)

    penalty4 = 0
    if constraint4 < 0:
        penalty4 = 1e6 * abs(constraint4)

    # Return negative power plus penalty
    return -W_elec + penalty1 + penalty2 + penalty3 + penalty4


def optimize_orc_de(orc_class, T_prod, p_surf, m_gf, T_am, T_reinj, V_d, A_eva_d, m_wf_d, A_cond_d,
                    bounds=None, constraints=None,
                    popsize=40, strategy='best1bin', maxiter=100, tol=0.01, mutation=(0.5, 1.0),
                    recombination=0.7, save_path=None, n_workers=None):
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(5e5, 30e5),  # p_pump (Pa)
                  (1, 15)]  # deltaT_ap_cond (K)

    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count() - 6  # Use all available CPU cores by default

    # Create a dictionary with all parameters for the objective function
    params_dict = {
        'T_prod': T_prod,
        'p_surf': p_surf,
        'm_gf': m_gf,
        'T_am': T_am,
        'T_reinj': T_reinj,
        'V_d': V_d,
        'A_eva_d': A_eva_d,
        'm_wf_d': m_wf_d,
        'A_cond_d': A_cond_d,
        'orc_class': orc_class
    }

    # Start timing the optimization
    start_time = time.time()

    # Run the differential evolution optimization with multiprocessing
    result = differential_evolution(
        func=single_objective,
        args=(params_dict,),  # Pass the parameters as a tuple with the dictionary
        bounds=bounds,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        disp=True,
        polish=False,  # Use a local optimizer at the end to refine the solution
        workers=n_workers,  # Pass the number of workers directly
        updating='deferred'  # Required for parallel processing
    )

    # End timing
    end_time = time.time()
    optimization_time = end_time - start_time

    # Get the optimal parameters
    optimal_params = result.x
    p_pump_opt, deltaT_ap_cond_opt = optimal_params

    # Create ORC with optimal parameters and calculate performance
    orc_optimal = orc_class(T_prod=T_prod,
                            p_surf=p_surf,
                            m_gf=m_gf,
                            T_ambient=T_am,
                            T_reinj=T_reinj,
                            V_d=V_d,
                            A_eva_d=A_eva_d,
                            m_wf_d=m_wf_d,
                            A_cond_d=A_cond_d,
                            p_pump=p_pump_opt,
                            deltaT_ap_cond=deltaT_ap_cond_opt)

    m_wf, W_elec, eta_ORC, exer_ORC, eta_t_off, T3, T_out_cool, m_cf, symb_eva, symb_cond, pinch = orc_optimal.ORC_results_geothermal()

    # Create results dictionary
    optimization_results = {
        'algorithm': 'Differential Evolution with Multiprocessing',
        'number_of_workers': n_workers,
        'working fluid': orc_optimal.fluid,
        'optimal_params': {
            'p_pump': float(p_pump_opt),
            'deltaT_ap_cond': float(deltaT_ap_cond_opt),
        },
        'electricity output (MW)': float(W_elec),  # MW
        'thermal efficiency': float(eta_ORC),
        'exergy efficiency': float(exer_ORC),
        'working fluid mass flow rate (kg/s)': float(m_wf),
        'turbine efficiency': float(eta_t_off),
        'turbine inlet temperature (C)': float(T3 - 273.15),
        'cooling fluid outlet temperature (C)': float(T_out_cool - 273.15),
        'cooling fluid mass flow rate (kg/s)': float(m_cf),
        'indicator of evaporator off-design': symb_eva,
        'indicator of condenser off-design': symb_cond,
        'pinch value of the evaporator during off-design': float(pinch),
        'de_parameters': {
            'popsize': popsize,
            'strategy': strategy,
            'maxiter': maxiter,
            'mutation': str(mutation),
            'recombination': recombination
        },
        'convergence_info': {
            'success': bool(result.success),
            'status': str(result.message),
            'nit': int(result.nit),
            'nfev': int(result.nfev)
        },
        'optimization_time (in second)': float(optimization_time)  # Time taken in seconds
    }

    # Save results to JSON file if path is provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(optimization_results, f, indent=4)
        print(f"Optimization results saved to {save_path}")

    # Print results
    print("\nDifferential Evolution Optimization Results:")
    print(f"Used {n_workers} CPU workers for parallel processing")
    print(f"Optimal pump pressure: {optimization_results['optimal_params']['p_pump'] / 1e5:.4f} bar")
    print(f"Optimal approach temperature difference: {optimization_results['optimal_params']['deltaT_ap_cond']:.4f} K")
    print(f"Maximum net electric power: {optimization_results['electricity output (MW)']:.5f} MW")
    print(f"Thermal efficiency: {optimization_results['thermal efficiency'] * 100:.2f}%")
    print(f"Exergy efficiency: {optimization_results['exergy efficiency'] * 100:.2f}%")
    print(f"Turbine efficiency: {optimization_results['turbine efficiency']:.3f}")
    print(f"Working fluid flow rate: {optimization_results['working fluid mass flow rate (kg/s)']:.2f} kg/s")
    print(f"Cooling fluid flow rate: {optimization_results['cooling fluid mass flow rate (kg/s)']:.2f} kg/s")
    print(f"Turbine inlet temperature: {optimization_results['turbine inlet temperature (C)']:.2f} C")
    print(f"Condenser outlet temperature: {optimization_results['cooling fluid outlet temperature (C)']:.2f} C")
    print(f"Optimization time: {optimization_results['optimization_time (in second)']:.2f} seconds")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Convergence status: {result.message}")

    return optimization_results


# Example usage:
if __name__ == "__main__":
    # This part ensures the multiprocessing starts correctly
    mp.freeze_support()  # Needed for Windows

    # Define known parameters
    T_prod = 120  # Production temperature in Celsius
    p_surf = 10  # Surface pressure in MPa
    m_gf = 50  # Mass flow rate of geothermal fluid in kg/s
    T_am = 20
    T_reinj = 70

    try:
        # Path to your JSON file
        file_path = "Design_optimization_results.json"

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        V_d = data['volumetric flow rate at design stage (m^3/s)']
        A_eva_d = data['surface area of the evaporator (m^2)']
        m_wf_d = data['working fluid mass flow rate (kg/s)']
        A_cond_d = data['surface area of the condenser (m^2)']

        # Run Differential Evolution optimization with multiprocessing
        # You can specify the number of workers or let it use all available cores
        results = optimize_orc_de(
            ORC, T_prod, p_surf, m_gf, T_am, T_reinj, V_d, A_eva_d, m_wf_d, A_cond_d,
            save_path=f"Off_design_optimization_results_{T_prod}.json",
            n_workers=60  # Specify desired number of cores or remove to use all available
        )
    except Exception as e:
        print(f"Error during optimization: {e}")