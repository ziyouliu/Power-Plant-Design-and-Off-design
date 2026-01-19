import numpy as np
from scipy.optimize import differential_evolution
import json
import time
import multiprocessing as mp
from Design_subcritical import ORC


# Define the objective function at the module level for picklability
def single_objective(x, params_dict):
    T_prod = params_dict['T_prod']
    p_surf = params_dict['p_surf']
    m_gf = params_dict['m_gf']
    T_am = params_dict['T_am']
    T_reinj = params_dict['T_reinj']
    orc_class = params_dict['orc_class']

    p_pump, deltaT_ap_cond, deltaT_pp_cond, deltaT_sh = x

    # Create ORC instance with the current parameters
    orc = orc_class(T_prod=T_prod,
                    p_surf=p_surf,
                    m_gf=m_gf,
                    T_ambient=T_am,
                    T_reinj=T_reinj,
                    p_pump=p_pump,
                    deltaT_ap_cond=deltaT_ap_cond,
                    deltaT_pp_cond=deltaT_pp_cond,
                    deltaT_sh=deltaT_sh)

    # Calculate ORC performance
    m_wf, W_elec, eta_ORC, exer_ORC, V, A_eva, pinch, A_cond, m_cf, sic = orc.ORC_results_geothermal()

    # Check constraint:
    constraint_value1 = orc.T_prod - orc.T3 - 5
    constraint_value2 = pinch - orc.min_pinch_eva
    constraint_value3 = orc.W_elec
    # constraint_value4 = orc.A_eva
    # constraint_value5 = orc.A_cond

    # Apply penalty if constraint is violated
    penalty1 = 0
    if constraint_value1 < 0:
        penalty1 = 1e6 * abs(constraint_value1)  # Large penalty for constraint violation

    penalty2 = 0
    if constraint_value2 < 0:
        penalty2 = 1e6 * abs(constraint_value2)

    penalty3 = 0
    if constraint_value3 < 0:
        penalty3 = 1e6 * abs(constraint_value3)

    # penalty4 = 0
    # if constraint_value4 < 0:
    #     penalty4 = 1e6 * abs(constraint_value4)
    #
    # penalty5 = 0
    # if constraint_value5 < 0:
    #     penalty5 = 1e6 * abs(constraint_value5)

    # Return negative power plus penalty
    return -W_elec + penalty1 + penalty2 + penalty3
    # return sic + penalty1 + penalty2 + penalty3


def optimize_orc_de(orc_class, T_prod, p_surf, m_gf, T_am, T_reinj, bounds=None, constraints=None,
                    popsize=80, strategy='best1bin', maxiter=100, tol=0.01, mutation=(0.5, 1.0),
                    recombination=0.7, save_path=None, n_workers=None):
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(10e5, 30e5),  # p_pump (Pa)
                  (6, 20),  # deltaT_ap_cond
                  (0.3, 5),  # deltaT_pp_cond
                  (0.01, 30)]  # deltaT_sh

    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()  # Use all available CPU cores by default

    # Create a dictionary with all parameters for the objective function
    params_dict = {
        'T_prod': T_prod,
        'p_surf': p_surf,
        'm_gf': m_gf,
        'T_am': T_am,
        'T_reinj': T_reinj,
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
    p_pump_opt, deltaT_ap_cond_opt, deltaT_pp_cond_opt, deltaT_sh_opt = optimal_params

    # Create ORC with optimal parameters and calculate performance
    orc_optimal = orc_class(T_prod=T_prod,
                            p_surf=p_surf,
                            m_gf=m_gf,
                            T_ambient=T_am,
                            T_reinj=T_reinj,
                            p_pump=p_pump_opt,
                            deltaT_ap_cond=deltaT_ap_cond_opt,
                            deltaT_pp_cond=deltaT_pp_cond_opt,
                            deltaT_sh=deltaT_sh_opt)

    m_wf, W_elec, eta_ORC, exer_ORC, V_d, A_eva_d, pinch_value_d, A_cond_d, m_cf, SIC = orc_optimal.ORC_results_geothermal()

    # Create results dictionary
    optimization_results = {
        'algorithm': 'Differential Evolution with Multiprocessing',
        'number_of_workers': n_workers,
        'working fluid': orc_optimal.fluid,
        'optimal_params': {
            'p_pump': float(p_pump_opt),
            'deltaT_ap_cond': float(deltaT_ap_cond_opt),
            'deltaT_pp_cond': float(deltaT_pp_cond_opt),
            'deltaT_sh': float(deltaT_sh_opt)
        },
        'electricity output (MW)': float(W_elec),  # MW
        'thermal efficiency': float(eta_ORC),
        'exergy efficiency': float(exer_ORC),
        'working fluid mass flow rate (kg/s)': float(m_wf),
        'volumetric flow rate at design stage (m^3/s)': float(V_d),
        'surface area of the evaporator (m^2)': float(A_eva_d),
        'pinch value of evaporator at design point': float(pinch_value_d),
        'surface area of the condenser (m^2)': float(A_cond_d),
        'air mass flow rate (kg/s):': float(m_cf),
        'SIC (kUSD/kW):': float(SIC),
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
    print(f"Optimal approach temperature difference of condenser: {optimization_results['optimal_params']['deltaT_ap_cond']:.4f} K")
    print(
        f"Optimal pinch point temperature difference of condenser: {optimization_results['optimal_params']['deltaT_pp_cond']:.4f} K")
    print(f"Optimal superheat degree: {optimization_results['optimal_params']['deltaT_sh']:.4f} K")
    print(f"Maximum net electric power: {optimization_results['electricity output (MW)']:.4f} MW")
    print(f"Thermal efficiency: {optimization_results['thermal efficiency'] * 100:.2f}%")
    print(f"Exergy efficiency: {optimization_results['exergy efficiency'] * 100:.2f}%")
    print(f"Working fluid flow rate: {optimization_results['working fluid mass flow rate (kg/s)']:.2f} kg/s")
    print(f"Surface area of the evaporator: {optimization_results['surface area of the evaporator (m^2)']:.2f} m^2")
    print(f"Pinch value of evaporator: {optimization_results['pinch value of evaporator at design point']}")
    print(f"Surface area of the condenser: {optimization_results['surface area of the condenser (m^2)']:.2f} m^2")
    print(f"Air mass flow rate: {optimization_results['air mass flow rate (kg/s):']} kg/s")
    print(f"SIC value: {optimization_results['SIC (kUSD/kW):']} kUSD/kW")
    print(f"Volumetric flow rate: {optimization_results['volumetric flow rate at design stage (m^3/s)']:.2f} m^3/s")
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
    T_prod = 150  # Production temperature in Celsius
    T_reinj = 70
    p_surf = 10  # Surface pressure in MPa
    m_gf = 50  # Mass flow rate of geothermal fluid in kg/s
    T_am = 20

    try:
        # Run Differential Evolution optimization with multiprocessing
        results = optimize_orc_de(
            ORC, T_prod, p_surf, m_gf, T_am, T_reinj,
            save_path="Design_optimization_results.json",
            n_workers=60  # Specify desired number of cores or remove to use all available
        )
    except Exception as e:
        print(f"Error during optimization: {e}")