import numpy as np
import jax
import jax.numpy as jnp
import timeit
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU

from utils import generate_material_distribution_indices
from utils import generate_material_list_with_air

from utils import generate_tmm_args, tmm_coh_tmm_wrapper
from utils import generate_vtmm_args
from utils import generate_tmm_fast_args
from utils import generate_tmmax_args

from tmmax.tmm import vectorized_coh_tmm
from vtmm import tmm_rt
from tmm_fast import coh_tmm


# Number of layers to test from 2 to 400 (exclusive), using integer values
number_of_layers = np.arange(2, 400, 20, dtype=int)

# Number of repetitions for the timeit function to get an average execution time
timeit_repetition = 50

# List of materials used in the simulations
material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]

# Polarization type, 's' polarization in this case
polarization = 's'

# Array of angles of incidence, linearly spaced from 0 to pi/2, with 20 points
angle_of_incidences = np.linspace(0, np.pi/2, 20)
angle_of_incidences_tmmax = jnp.array(angle_of_incidences)

# Wavelength array from 500 nm to 1000 nm, linearly spaced with 20 points
wavelength_arr = np.linspace(500e-9, 1000e-9, 20)
wavelength_arr_tmmax = jnp.array(wavelength_arr)

# Lists to store execution times for each method
time_tmm = []
time_vtmm = []
time_tmm_fast = []
time_tmmax = []

# Loop through different numbers of layers
for N in number_of_layers:
    
    # Generate random material distribution indices and the corresponding material list with air layers
    indices = generate_material_distribution_indices(N, low=0, high=len(material_set))
    material_list = generate_material_list_with_air(indices, material_set) # np.load(f"material_distribution_with_layer_num_{N}.npy", allow_pickle=True)
    
    # Save the material distribution list as a .npy file for reference
    np.save(f"material_distribution_with_layer_num_{N}.npy", material_list)
    
    # Randomly generate thicknesses for each layer between 100 nm and 500 nm
    thickness_list = np.random.uniform(100, 500, N) * 1e-9 # np.load(f"thickness_list_with_layer_num_{N}.npy")
    np.save(f"thickness_list_with_layer_num_{N}.npy", thickness_list)  # Save the thickness list

    thickness_list_tmm, nk_list_tmm = generate_tmm_args(material_list = material_list, 
                                                    thickness_list = thickness_list,
                                                    wavelength_arr = wavelength_arr)
    
    omega_vtmm, kx_vtmm, nk_list_vtmm, thickness_list_vtmm = generate_vtmm_args(wavelength_arr = wavelength_arr, 
                                                                                angle_of_incidences = angle_of_incidences,
                                                                                thickness_list = thickness_list,
                                                                                material_list = material_list)
    
    t_tmm = timeit.timeit( lambda: tmm_coh_tmm_wrapper(polarization, nk_list_tmm, thickness_list_tmm, angle_of_incidences, wavelength_arr),  number=timeit_repetition )
    t_vtmm = timeit.timeit( lambda: tmm_rt(polarization, omega_vtmm, kx_vtmm, nk_list_vtmm, thickness_list_vtmm),  number = timeit_repetition )

    M_tmm_fast, T_tmm_fast = generate_tmm_fast_args(material_list = material_list, 
                                                    thickness_list = thickness_list,
                                                    wavelength_arr = wavelength_arr)

    t_tmm_fast = timeit.timeit( lambda: coh_tmm(polarization, M_tmm_fast, T_tmm_fast, angle_of_incidences, wavelength_arr, device='cpu'),  number = timeit_repetition )


    thickness_list_tmmax = jnp.array(thickness_list)
    
    data_tmmax, material_distribution_tmmax, polarization_tmmax = generate_tmmax_args(material_list = material_list, polarization = polarization)
    
    t_tmmax = timeit.timeit( lambda: jax.block_until_ready(vectorized_coh_tmm(data_tmmax, material_distribution_tmmax, thickness_list_tmmax, wavelength_arr_tmmax, angle_of_incidences_tmmax, polarization_tmmax)),  number = timeit_repetition)
    
    time_tmm.append(t_tmm)
    time_vtmm.append(t_vtmm)
    time_tmm_fast.append(t_tmm_fast)
    time_tmmax.append(t_tmmax)



    print(f"{N} tmm took ", t_tmm)
    print(f"{N} vtmm took ", t_vtmm)
    print(f"{N} tmm-fast took ", t_tmm_fast)
    print(f"{N} tmmax took ", t_tmmax)

# Save the time measurements for each method into .npy files
np.save("layer_count_exp_time_of_tmm.npy", time_tmm)
np.save("layer_count_exp_time_of_vtmm.npy", time_vtmm)
np.save("layer_count_exp_time_of_tmm_fast.npy", time_tmm_fast)
np.save("layer_count_exp_time_of_tmmax.npy", time_tmmax)