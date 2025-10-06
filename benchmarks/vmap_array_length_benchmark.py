import numpy as np
import jax
import jax.numpy as jnp
import timeit

from utils import generate_material_distribution_indices
from utils import generate_material_list_with_air

from utils import generate_tmm_args, tmm_coh_tmm_wrapper
from utils import generate_vtmm_args
from utils import generate_tmm_fast_args
from utils import generate_tmmax_args

from tmmax.tmm import vectorized_coh_tmm
from vtmm import tmm_rt
from tmm_fast import coh_tmm

wl_arr_lengths = np.arange(2, 100, 5, dtype = int)
angle_of_incidences_arr_lengths = np.arange(2, 100, 5, dtype = int)
timeit_repetition = 50
num_of_layers = 8 # or 80

material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]
polarization = 's'

indices = generate_material_distribution_indices(num_of_layers, low=0, high=len(material_set))
material_list = generate_material_list_with_air(indices, material_set) # np.load(f"material_distribution_with_layer_num_{num_of_layers}_wl_arr_exp.npy", allow_pickle=True)
np.save(f"material_distribution_with_layer_num_{num_of_layers}_wl_arr_exp.npy", material_list)

thickness_list  = np.random.uniform(100, 500, num_of_layers)*1e-9 # np.load(f"thickness_list_with_layer_num_{num_of_layers}_wl_arr_exp.npy")
thickness_list_tmmax = jnp.array(thickness_list)
np.save(f"thickness_list_with_layer_num_{num_of_layers}_wl_arr_exp.npy", thickness_list)

time_tmm = np.zeros((len(wl_arr_lengths), len(angle_of_incidences_arr_lengths))) 
time_vtmm = np.zeros((len(wl_arr_lengths), len(angle_of_incidences_arr_lengths)))
time_tmm_fast = np.zeros((len(wl_arr_lengths), len(angle_of_incidences_arr_lengths)))
time_tmmax = np.zeros((len(wl_arr_lengths), len(angle_of_incidences_arr_lengths))) 

for i in range(len(wl_arr_lengths)):
    for j in range(len(angle_of_incidences_arr_lengths)):
        
        wl_arr_length = wl_arr_lengths[i]
        angle_of_incidences_arr_length = angle_of_incidences_arr_lengths[j]

        wavelength_arr = np.linspace(500e-9, 1000e-9, wl_arr_length)
        wavelength_arr_tmmax = jnp.array(wavelength_arr)

        angle_of_incidences = np.linspace(0, np.pi/2, angle_of_incidences_arr_length)
        angle_of_incidences_tmmax = jnp.array(angle_of_incidences)

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
        
        time_tmm[i,j] = t_tmm
        time_vtmm[i,j] = t_vtmm
        time_tmm_fast[i,j] = t_tmm_fast
        time_tmmax[i,j] = t_tmmax

        print(f"{wl_arr_length}, {angle_of_incidences_arr_length} tmm took ", t_tmm)
        print(f"{wl_arr_length}, {angle_of_incidences_arr_length} vtmm took ", t_vtmm)
        print(f"{wl_arr_length}, {angle_of_incidences_arr_length} tmm fast took ", t_tmm_fast)
        print(f"{wl_arr_length}, {angle_of_incidences_arr_length} tmmax took ", t_tmmax)


np.save("time_of_tmm_wl_theta_arr.npy", time_tmm)
np.save("time_of_vtmm_wl_theta_arr.npy", time_vtmm)
np.save("time_of_tmm_fast_wl_theta_arr.npy", time_tmm_fast)
np.save("time_of_tmmax_wl_theta_arr.npy", time_tmmax)