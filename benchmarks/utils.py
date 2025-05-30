import numpy as np
import jax.numpy as jnp
import tensorflow as tf

from tmm import coh_tmm
from tmmax.data import interpolate_nk, material_distribution_to_set, create_data

from typing import List, Callable, Tuple, Union

def get_nk_values(wl: float, nk_functions: List[Callable[[float], complex]], material_list: List[int]) -> np.ndarray:
    """
    This function retrieves the refractive index and extinction coefficient values 
    (represented as complex numbers) for a given wavelength `wl` from a list of materials.
    Each material has a corresponding function in the `nk_functions` list that, 
    when provided with the wavelength, returns the refractive index (n) and extinction 
    coefficient (k) as a complex number (n + j*k). The materials are indexed 
    by `material_list` to access their corresponding functions.

    Args:
    wl (float): The wavelength at which to compute the refractive index and extinction coefficient.
    nk_functions (List[Callable[[float], complex]]): A list of functions, each corresponding to 
    a material's refractive index and extinction coefficient. These functions take the wavelength 
    as input and return a complex number (n + j*k).
    material_list (List[int]): A list of indices, where each index corresponds to a material, 
    and is used to retrieve the respective function from the `nk_functions` list.

    Returns:
    np.ndarray: An array of complex numbers where each entry corresponds to the refractive index (n) 
    and extinction coefficient (k) of a material at the given wavelength `wl`.

    """
    
    return np.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Convert the resulting list to a NumPy array.

def generate_tmm_args(material_list: List[str], 
                      thickness_list: Union[np.ndarray, float],
                      wavelength_arr: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function calculates the reflection (R) and transmission (T) for a multilayer thin film stack 
    over arrays of wavelengths and angles of incidence using the coherent Transfer Matrix Method (TMM). 
    The polarization of the incident light is considered either "s" (TE) or "p" (TM). The function 
    interpolates the refractive index (n) and extinction coefficient (k) for the materials at given 
    wavelengths and applies TMM over the material layers for each wavelength and angle of incidence.
    
    Args:
        polarization (str): Polarization of the incident light. Either "s" (TE) or "p" (TM).
        material_list (List[str]): List of material names for each layer.
        thickness_list (Union[np.ndarray, float]): Array of thicknesses for each layer in nanometers.
        angle_of_incidences (Union[np.ndarray, float]): Array of angles of incidence in degrees.
        wavelength_arr (Union[np.ndarray, float]): Array of wavelengths in nanometers.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays for reflection (R) and transmission (T). 
        The shape of these arrays will be (len(wavelength_arr), len(angle_of_incidences)).
    """

    # Create a set of unique materials to avoid redundant interpolation # The list(set(...)) ensures unique materials.
    material_set = list(set(material_list))  
    
    # Assign each unique material an enumerated integer value to map it efficiently later # A dictionary with material names as keys and unique indices as values.
    material_enum = {material: i for i, material in enumerate(material_set)}
    
    # Replace material names in the list with their enumerated integer index using the dictionary created # Converts material names in material_list to their corresponding integer identifiers.
    material_list = [int(material_enum[material]) for material in material_list]
    
    # Create a dictionary mapping the enumerated material indices to their interpolated nk functions # Prepares interpolation functions for each unique material's refractive index data.
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}
    
    # Extend the thickness list by adding infinite boundaries for air above and below the stack # np.inf ensures that the first and last "layers" are considered infinitely thick (air layers).
    thickness_list = np.concatenate(([np.inf], thickness_list, [np.inf]), axis=None)

    nk_list = get_nk_values(wavelength_arr, nk_funkcs, material_list)

    return thickness_list, nk_list

def tmm_coh_tmm_wrapper(polarization: str, 
                        nk_list: np.ndarray, 
                        thickness_list: np.ndarray, 
                        angle_of_incidences: np.ndarray, 
                        wavelength_arr: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        polarization (str): Polarization of the incident light. Either "s" (TE) or "p" (TM).
        nk_list (List[str]): List of material names for each layer.
        thickness_list (Union[np.ndarray, float]): Array of thicknesses for each layer in nanometers.
        angle_of_incidences (Union[np.ndarray, float]): Array of angles of incidence in degrees.
        wavelength_arr (Union[np.ndarray, float]): Array of wavelengths in nanometers.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays for reflection (R) and transmission (T). 
        The shape of these arrays will be (len(wavelength_arr), len(angle_of_incidences)).
    """
    # Initialize empty arrays for storing reflection (R) and transmission (T) results # The result arrays have dimensions len(wavelength_arr) x len(angle_of_incidences).
    R = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    T = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    
    # Nested loops to compute R and T for each combination of wavelength and angle of incidence # Outer loop: iterating over wavelengths; Inner loop: iterating over angles of incidence.
    for i in range(len(wavelength_arr)):
        for j in range(len(angle_of_incidences)):
            
            # Retrieve the refractive index (n) and extinction coefficient (k) for each material at the current wavelength # nk_list contains n and k for all materials at wavelength_arr[i].
            one_wl_nk_list = nk_list[:,i]
            
            # Perform the coherent TMM calculation using the polarization, nk_list, thicknesses, and current angle/wavelength # The result is a dictionary containing 'R' and 'T'.
            result = coh_tmm(polarization, one_wl_nk_list, thickness_list, angle_of_incidences[j], wavelength_arr[i])
            
            # Store the calculated reflection (R) and transmission (T) in the result arrays # Assign reflection and transmission values to the corresponding index in R and T arrays.
            R[i,j] = result['R']
            T[i,j] = result['T']

    # Return the final reflection (R) and transmission (T) arrays for all wavelengths and angles # These are 2D arrays, where each element corresponds to a specific wavelength and angle.
    return R, T

def generate_vtmm_args(wavelength_arr: np.ndarray, 
                       angle_of_incidences: np.ndarray, 
                       thickness_list: np.ndarray, 
                       material_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates and prepares arguments required for the vtmm.
    
    Arguments:
    ----------
    wavelength_arr : np.ndarray
        An array of wavelengths (in meters) for which calculations will be performed.
    angle_of_incidences : np.ndarray
        An array of angles of incidence (in radians) corresponding to the wavelengths.
    thickness_list : np.ndarray
        An array representing the thicknesses of different layers in the multilayer system.
    material_list : List[str]
        A list of material names corresponding to each layer in the system.

    Returns:
    --------
    omega : np.ndarray
        Angular frequencies (omega) computed from the input wavelength array.
    kx : np.ndarray
        The parallel component of the wavevector computed from the angles of incidence and wavelengths.
    nk_list : np.ndarray
        The refractive index and extinction coefficient values for each material at the first wavelength.
    thickness_list : np.ndarray
        The layer thicknesses, converted to TensorFlow tensors for further computations.
    """

    speed_of_light = 299792458 # m/s
    
    # Create a unique set of materials from the input material list to avoid duplicates
    material_set = list(set(material_list))  # Convert material list to set for unique materials
    
    # Create an enumeration mapping of materials to integers for easier indexing in future calculations
    material_enum = {material: i for i, material in enumerate(material_set)}  # Material to index dictionary
    
    # Replace material names in the material list with their corresponding numerical indices
    material_list = [int(material_enum[material]) for material in material_list]  # List of material indices
    
    # Retrieve interpolating functions for refractive index and extinction coefficient for each material
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}  # Functions for n and k values
    
    # Get the refractive index and extinction coefficient values for the first wavelength in the array
    # Using nk_funkcs, this retrieves n and k values for all materials at the specified wavelength
    nk_list = get_nk_values(wavelength_arr[0], nk_funkcs, material_list)  # n and k values for the materials
    nk_list = tf.cast(tf.convert_to_tensor(nk_list), tf.float64) # Convert n and k values to a TensorFlow tensor
    
    # Calculate angular frequency (omega) from the wavelength array. Omega = 2 * pi * c / wavelength
    omega = speed_of_light / wavelength_arr * 2 * np.pi  # Convert wavelength to angular frequency
    omega = tf.cast(tf.convert_to_tensor(omega), tf.float64) # Convert omega to a TensorFlow tensor
    
    
    # Calculate the parallel component of the wavevector (kx) from the angle of incidence and wavelength
    kx = np.sin(angle_of_incidences) * 2 * np.pi / wavelength_arr[0]  # Calculate kx from angles and wavelength
    kx = tf.cast(tf.convert_to_tensor(kx), tf.float64) # Convert kx to a TensorFlow tensor
    
    kx = tf.constant(kx) # Ensure kx is a TensorFlow constant
    
    thickness_list = tf.cast(tf.convert_to_tensor(thickness_list), tf.float64) # Convert thicknesses to TensorFlow tensor
    
    # Return the calculated parameters required for vtmm function
    return omega, kx, nk_list, thickness_list

def generate_tmmax_args(material_list: List[str], polarization: str):
    """
    Generates the required arguments for tmmax function

    Arguments:
    - material_list (List[str]): A list of material names defining the structure.
    - polarization (str): The polarization type, which must be either 's' or 'p'.

    Returns:
    - data (jnp.ndarray): A jax array containing optical properties (such as refractive index and extinction coefficient)
      for each unique material in the input list. This data is necessary for performing tmmax calculations.
    - material_distribution (List[str]): A processed list that maintains the original material sequence,
      ensuring correct layer stacking in the simulation.
    - polarization (jnp.ndarray): A JAX array of shape (1,) with a boolean value.
      - `False` represents s-polarization.
      - `True` represents p-polarization.
    
    Raises:
    - TypeError: If the input polarization string is not 's' or 'p'.
    """

    # Convert material_list into a unique material set and maintain its distribution order.
    material_set, material_distribution = material_distribution_to_set(material_list)
    
    # Create the required material data, such as refractive index and extinction coefficient, for each material in the set.
    # This function retrieves optical properties needed for tmmax calculations.
    data = create_data(material_set)

    # Check the polarization input and convert it to a boolean JAX array.
    if polarization == 's':
        # For s-polarization, set the boolean flag to `False`.
        polarization = jnp.array([False], dtype=bool)
    elif polarization == 'p':
        # For p-polarization, set the boolean flag to `True`.
        polarization = jnp.array([True], dtype=bool)
    else:
        # Raise an error if the polarization input is invalid.
        raise TypeError("The polarization can be 's' or 'p', not other parts. Correct it")

    return data, material_distribution, polarization


def generate_tmm_fast_args(material_list: List[str], 
                           thickness_list: Union[np.ndarray, float],
                           wavelength_arr: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the necessary arguments for a tmm-fast simulation in an optimized manner.
    
    Arguments:
    ----------
    material_list : List[str]
        A list of material names (strings) corresponding to each layer in the stack.
        
    thickness_list : Union[np.ndarray, float]
        An array (or a single float) representing the thickness of each material layer.
        
    wavelength_arr : Union[np.ndarray, float]
        An array (or a single float) representing the wavelength values for the simulation.
        
    Returns:
    --------
    M : np.ndarray
        A complex-valued 3D array of shape (1, num_layers, num_wavelengths) containing the refractive 
        index values for each layer at each wavelength. 
        
    T : np.ndarray
        A 2D array of shape (1, num_layers) containing the thickness values for each layer, including 
        the added infinite air boundaries.
    """

    # Create a set of unique materials to avoid redundant interpolation
    # The list(set(...)) ensures unique materials are stored only once.
    material_set = list(set(material_list))  
    
    # Assign each unique material an enumerated integer value to map it efficiently later
    # A dictionary with material names as keys and unique indices as values.
    material_enum = {material: i for i, material in enumerate(material_set)}
    
    # Replace material names in the list with their enumerated integer index using the dictionary created
    # Converts material names in material_list to their corresponding integer identifiers.
    material_list = [int(material_enum[material]) for material in material_list]
    
    # Create a dictionary mapping the enumerated material indices to their interpolated nk functions
    # Each function is responsible for interpolating the refractive index at a given wavelength.
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}
    
    # Extend the thickness list by adding infinite boundaries for air above and below the stack
    # np.inf ensures that the first and last "layers" are considered infinitely thick (air layers).
    thickness_list = np.concatenate(([np.inf], thickness_list, [np.inf]), axis=None)

    # Compute the refractive index values for each material at the given wavelengths
    nk_list = get_nk_values(wavelength_arr, nk_funkcs, material_list)

    # Initialize an array M with shape (1, num_layers, num_wavelengths) to store refractive index values
    # The dtype is set to np.complex128 to account for materials with non-zero extinction coefficients.
    M = np.ones((1, len(thickness_list), len(wavelength_arr)), dtype=np.complex128)

    # Assign the computed refractive index values to M
    M[0,:,:] = nk_list

    # Initialize an array T with shape (1, num_layers) to store thickness values
    T = np.zeros((1, len(thickness_list)))

    # Assign the computed thickness values (including air boundaries) to T
    T[0,:] = thickness_list
    
    return M, T

def generate_material_distribution_indices(N, low=0, high=10):
    """
    Generates an array of random integers with length N such that
    no two consecutive elements are the same.

    Parameters:
        N (int): Length of the array.
        low (int): Minimum value of the integers (inclusive).
        high (int): Maximum value of the integers (exclusive).

    Returns:
        numpy.ndarray: Array of random integers with no consecutive elements the same.
    """
    if N <= 0:
        raise ValueError("Array length N must be positive.")
    
    arr = np.zeros(N, dtype=int)
    arr[0] = np.random.randint(low, high)
    
    for i in range(1, N):
        prev = arr[i-1]
        new_val = prev
        while new_val == prev:
            new_val = np.random.randint(low, high)
        arr[i] = new_val

    return arr

def generate_material_list_with_air(index_array, material_list):
    """
    Takes an array of indices and a material list, and generates a list of materials
    based on the index array, with 'Air' concatenated at the start and end.

    Parameters:
        index_array (numpy.ndarray): Array of integers representing indices.
        material_list (list of str): List of material names (strings).

    Returns:
        list of str: List of materials with 'Air' at the start and end.
    """
    if not all(0 <= idx < len(material_list) for idx in index_array):
        raise ValueError("Index out of bounds for the material list.")
    
    # Generate the list of materials
    material_sequence = [material_list[idx] for idx in index_array]
    
    # Concatenate "Air" at the start and end
    return ["Air"] + material_sequence + ["Air"]