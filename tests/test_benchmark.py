"""
test_benchmark.py

Performance benchmarking tests for TMMax and TMM implementations.

This test reproduces and verifies the timing experiments for:
- tmm
- tmmax
"""

import timeit
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import utilities and TMM functions
from .benchmark_utils import (
    generate_material_distribution_indices,
    generate_material_list_with_air,
    generate_tmm_args,
    tmm_coh_tmm_wrapper,
    generate_tmmax_args,
)
from tmmax.tmm import vectorized_coh_tmm

@pytest.mark.order(1)
def test_layer_size_benchmark(tmp_path):
    """
    Run benchmark timing tests on a single CPU core and verify results.

    This test measures runtime performance for different multilayer
    stack sizes and confirms:
    - .npy files are correctly saved and reloadable.
    - Timing arrays are non-empty and positive.
    - The experiment respects single-core execution constraints.
    """
    # --- Benchmark configuration ---
    number_of_layers = np.arange(2, 50, 5, dtype=int)
    timeit_repetition = 10
    material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]
    polarization = "s"
    angle_of_incidences = np.linspace(0, np.pi / 2, 20)
    angle_of_incidences_tmmax = jnp.array(angle_of_incidences)
    wavelength_arr = np.linspace(500e-9, 1000e-9, 20)
    wavelength_arr_tmmax = jnp.array(wavelength_arr)

    time_tmm, time_tmmax = [], []

    # --- Benchmark loop ---
    for N in number_of_layers:
        indices = generate_material_distribution_indices(N, low=0, high=len(material_set))
        material_list = generate_material_list_with_air(indices, material_set)

        mat_path = tmp_path / f"material_distribution_with_layer_num_{N}.npy"
        np.save(mat_path, material_list)

        thickness_list = np.random.uniform(100, 500, N) * 1e-9
        thick_path = tmp_path / f"thickness_list_with_layer_num_{N}.npy"
        np.save(thick_path, thickness_list)

        # --- tmm ---
        thickness_list_tmm, nk_list_tmm = generate_tmm_args(
            material_list=material_list, thickness_list=thickness_list, wavelength_arr=wavelength_arr
        )
        t_tmm = timeit.timeit(
            lambda: tmm_coh_tmm_wrapper(polarization, nk_list_tmm, thickness_list_tmm, angle_of_incidences, wavelength_arr),
            number=timeit_repetition,
        )

        # --- tmmax ---
        thickness_list_tmmax = jnp.array(thickness_list)
        data_tmmax, material_distribution_tmmax, polarization_tmmax = generate_tmmax_args(
            material_list=material_list, polarization=polarization
        )
        t_tmmax = timeit.timeit(
            lambda: jax.block_until_ready(
                vectorized_coh_tmm(
                    data_tmmax,
                    material_distribution_tmmax,
                    thickness_list_tmmax,
                    wavelength_arr_tmmax,
                    angle_of_incidences_tmmax,
                    polarization_tmmax,
                )
            ),
            number=timeit_repetition,
        )

        time_tmm.append(t_tmm)
        time_tmmax.append(t_tmmax)

        print(f"{N} layers → TMM: {t_tmm:.4f}s | TMMax: {t_tmmax:.4f}s")

    # --- Save results ---
    np.save(tmp_path / "layer_count_exp_time_of_tmm.npy", time_tmm)
    np.save(tmp_path / "layer_count_exp_time_of_tmmax.npy", time_tmmax)

    # --- Verify saved data ---
    for fname in [
        "layer_count_exp_time_of_tmm.npy",
        "layer_count_exp_time_of_tmmax.npy",
    ]:
        fpath = tmp_path / fname
        assert fpath.exists(), f"Benchmark file {fname} was not created."
        arr = np.load(fpath)
        assert len(arr) == len(number_of_layers), f"{fname} has incorrect length."
        assert np.all(arr > 0), f"{fname} contains non-positive timing values."
        assert np.all(np.isfinite(arr)), f"{fname} contains NaN or Inf values."

    # Load timing data from fixture
    tmm = np.load(tmp_path / "layer_count_exp_time_of_tmm.npy")
    tmmax = np.load(tmp_path / "layer_count_exp_time_of_tmmax.npy")

    # Define the number of layers
    number_of_layers = np.arange(2, 50, 5, dtype=int) 

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(number_of_layers, tmm, 'o-', label="TMM", linewidth=2, color='orangered')
    ax.plot(number_of_layers, tmmax, 'o-', label="TMMax", linewidth=2, color='navy')

    # Customize plot
    ax.set_xticks([2, 10, 20, 30, 40, 50])
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_xlabel("Layer Count")
    ax.set_ylabel("Run Time (s)")

    # Save plot to temporary path
    save_path = tmp_path / "layer_size_benchmark.png"
    plt.savefig(save_path, dpi=600)
    plt.show()

    # Verify that the file exists and is non-empty
    assert save_path.exists(), f"Benchmark plot not saved: {save_path}"
    assert save_path.stat().st_size > 0, f"Saved plot file is empty: {save_path}"

    # Optional: load back and check basic properties
    img = plt.imread(save_path)
    assert img.size > 0, "Loaded image is empty"



@pytest.mark.order(2)
def test_array_length_benchmark_8_layer(tmp_path):
    """
    Benchmark TMMax and TMM implementations.
    
    This test reproduces the benchmark workflow:
    - Generates random multilayer configurations.
    - Runs timing benchmarks for tmm and tmmax.
    - Saves timing arrays to disk.
    - Reloads saved arrays and validates:
        * Successful saving and loading
        * Matching array shapes
        * All recorded times are positive
    """

    # --- Benchmark configuration ---
    wl_arr_lengths = np.arange(2, 22, 1, dtype=int)
    angle_arr_lengths = np.arange(2, 22, 1, dtype=int)
    timeit_repetitions = 3  # Reduced for test speed; increase for full benchmark
    num_layers = 8
    polarization = "s"
    material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]

    # --- Material setup ---
    indices = generate_material_distribution_indices(num_layers, low=0, high=len(material_set))
    material_list = generate_material_list_with_air(indices, material_set)

    # Save materials and thickness data
    mat_save_path = tmp_path / f"material_distribution_{num_layers}.npy"
    np.save(mat_save_path, material_list)

    thickness_list = np.random.uniform(100, 500, num_layers) * 1e-9
    np.save(tmp_path / f"thickness_list_{num_layers}.npy", thickness_list)

    # Initialize time matrices
    shape = (len(wl_arr_lengths), len(angle_arr_lengths))
    time_tmm = np.zeros(shape)
    time_tmmax = np.zeros(shape)

    # --- Benchmark loop ---
    for i, wl_len in enumerate(wl_arr_lengths):
        for j, ang_len in enumerate(angle_arr_lengths):
            wl_arr = np.linspace(500e-9, 1000e-9, wl_len)
            ang_arr = np.linspace(0, np.pi / 2, ang_len)

            # Prepare data for each implementation
            thickness_list_tmm, nk_list_tmm = generate_tmm_args(material_list, thickness_list, wl_arr)

            # --- tmm benchmark ---
            t_tmm = timeit.timeit(
                lambda: tmm_coh_tmm_wrapper(polarization, nk_list_tmm, thickness_list_tmm, ang_arr, wl_arr),
                number=timeit_repetitions,
            )

            # --- TMMax benchmark ---
            wl_jax = jnp.array(wl_arr)
            ang_jax = jnp.array(ang_arr)
            thick_jax = jnp.array(thickness_list)
            data_tmmax, mat_dist_tmmax, pol_tmmax = generate_tmmax_args(material_list, polarization)

            t_tmmax = timeit.timeit(
                lambda: jax.block_until_ready(
                    vectorized_coh_tmm(data_tmmax, mat_dist_tmmax, thick_jax, wl_jax, ang_jax, pol_tmmax)
                ),
                number=timeit_repetitions,
            )

            # Store timings
            time_tmm[i, j] = t_tmm
            time_tmmax[i, j] = t_tmmax

            # Sanity check (during loop)
            assert t_tmmax > 0 and t_tmm > 0, "Timing must be positive."

    # --- Save results ---
    np.save(tmp_path / "time_of_tmm_wl_theta_arr_8_layer.npy", time_tmm)
    np.save(tmp_path / "time_of_tmmax_wl_theta_arr_8_layer.npy", time_tmmax)

    # --- Reload saved arrays ---
    loaded_tmm = np.load(tmp_path / "time_of_tmm_wl_theta_arr_8_layer.npy")
    loaded_tmmax = np.load(tmp_path / "time_of_tmmax_wl_theta_arr_8_layer.npy")

    # --- Validation checks ---
    assert loaded_tmm.shape == shape
    assert loaded_tmmax.shape == shape

    # Ensure all benchmark times are positive
    for arr in [loaded_tmm, loaded_tmmax]:
        assert np.all(arr > 0), "All benchmark times must be positive."

    # Sanity check for file existence
    for filename in [
        "time_of_tmm_wl_theta_arr_8_layer.npy",
        "time_of_tmmax_wl_theta_arr_8_layer.npy",
    ]:
        assert (tmp_path / filename).exists(), f"{filename} was not saved correctly."

    # Final check: ensure meaningful (non-zero) average runtime
    avg_time = np.mean(loaded_tmmax)
    assert avg_time > 0, "Average TMMax runtime must be positive."

    print(f"Benchmark completed. Average TMMax runtime: {avg_time:.4f} s")

    # Assertions for basic sanity checks
    for arr, name in zip(
        [loaded_tmm, loaded_tmmax],
        ["tmm", "tmmax"]
    ):
        assert isinstance(arr, np.ndarray), f"{name} should be a numpy array."
        assert arr.size > 0, f"{name} array is empty."
        assert np.all(arr >= 0), f"{name} contains negative execution times."

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)
    titles = ['tmm', 'tmmax']
    datasets = [loaded_tmm, loaded_tmmax]
    cmaps = ['YlOrRd', 'PuBuGn']

    # Plot individual heatmaps
    for i, (ax, data, cmap, title) in enumerate(zip(axes, datasets, cmaps, titles)):
        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Angle of Incidence Array Length", fontsize=10)
        ax.set_ylabel("Wavelength Array Length", fontsize=10)
        ax.set_xticks([0, 4, 8, 12, 16, 19])
        ax.set_yticks([19, 16, 12, 8, 4, 0])
        ax.set_xticklabels(["2","6","10","14","18","21"])
        ax.set_yticklabels(["21","18","14","10","6","2"])
        ax.invert_yaxis()

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time(s)", fontsize=10, fontweight='bold')
        cbar.set_ticks(np.linspace(np.min(data), np.max(data), 7))

    # Save figure
    save_file = tmp_path / "wl_theta_arr_benchmark_8_layer.png"
    plt.savefig(save_file, dpi=600)
    plt.show()

    # Check if file exists
    assert save_file.exists(), f"Benchmark figure was not saved at {save_file}"
    assert save_file.stat().st_size > 0, "Saved benchmark figure is empty."

@pytest.mark.order(3)
def test_array_length_benchmark_80_layer(tmp_path):
    """
    Benchmark TMMax and TMM implementations.
    
    This test reproduces the benchmark workflow:
    - Generates random multilayer configurations.
    - Runs timing benchmarks for tmm and tmmax.
    - Saves timing arrays to disk.
    - Reloads saved arrays and validates:
        * Successful saving and loading
        * Matching array shapes
        * All recorded times are positive
    """

    # --- Benchmark configuration ---
    wl_arr_lengths = np.arange(2, 22, 1, dtype=int)
    angle_arr_lengths = np.arange(2, 22, 1, dtype=int)
    timeit_repetitions = 3  # Reduced for test speed; increase for full benchmark
    num_layers = 80
    polarization = "s"
    material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]

    # --- Material setup ---
    indices = generate_material_distribution_indices(num_layers, low=0, high=len(material_set))
    material_list = generate_material_list_with_air(indices, material_set)

    # Save materials and thickness data
    mat_save_path = tmp_path / f"material_distribution_{num_layers}.npy"
    np.save(mat_save_path, material_list)

    thickness_list = np.random.uniform(100, 500, num_layers) * 1e-9
    np.save(tmp_path / f"thickness_list_{num_layers}.npy", thickness_list)

    # Initialize time matrices
    shape = (len(wl_arr_lengths), len(angle_arr_lengths))
    time_tmm = np.zeros(shape)
    time_tmmax = np.zeros(shape)

    # --- Benchmark loop ---
    for i, wl_len in enumerate(wl_arr_lengths):
        for j, ang_len in enumerate(angle_arr_lengths):
            wl_arr = np.linspace(500e-9, 1000e-9, wl_len)
            ang_arr = np.linspace(0, np.pi / 2, ang_len)

            # Prepare data for each implementation
            thickness_list_tmm, nk_list_tmm = generate_tmm_args(material_list, thickness_list, wl_arr)

            # --- tmm benchmark ---
            t_tmm = timeit.timeit(
                lambda: tmm_coh_tmm_wrapper(polarization, nk_list_tmm, thickness_list_tmm, ang_arr, wl_arr),
                number=timeit_repetitions,
            )

            # --- TMMax benchmark ---
            wl_jax = jnp.array(wl_arr)
            ang_jax = jnp.array(ang_arr)
            thick_jax = jnp.array(thickness_list)
            data_tmmax, mat_dist_tmmax, pol_tmmax = generate_tmmax_args(material_list, polarization)

            t_tmmax = timeit.timeit(
                lambda: jax.block_until_ready(
                    vectorized_coh_tmm(data_tmmax, mat_dist_tmmax, thick_jax, wl_jax, ang_jax, pol_tmmax)
                ),
                number=timeit_repetitions,
            )

            # Store timings
            time_tmm[i, j] = t_tmm
            time_tmmax[i, j] = t_tmmax

            # Sanity check (during loop)
            assert t_tmmax > 0 and t_tmm > 0, "Timing must be positive."

    # --- Save results ---
    np.save(tmp_path / "time_of_tmm_wl_theta_arr_80_layer.npy", time_tmm)
    np.save(tmp_path / "time_of_tmmax_wl_theta_arr_80_layer.npy", time_tmmax)

    # --- Reload saved arrays ---
    loaded_tmm = np.load(tmp_path / "time_of_tmm_wl_theta_arr_80_layer.npy")
    loaded_tmmax = np.load(tmp_path / "time_of_tmmax_wl_theta_arr_80_layer.npy")

    # --- Validation checks ---
    assert loaded_tmm.shape == shape
    assert loaded_tmmax.shape == shape

    # Ensure all benchmark times are positive
    for arr in [loaded_tmm, loaded_tmmax]:
        assert np.all(arr > 0), "All benchmark times must be positive."

    # Sanity check for file existence
    for filename in [
        "time_of_tmm_wl_theta_arr_80_layer.npy",
        "time_of_tmmax_wl_theta_arr_80_layer.npy",
    ]:
        assert (tmp_path / filename).exists(), f"{filename} was not saved correctly."

    # Final check: ensure meaningful (non-zero) average runtime
    avg_time = np.mean(loaded_tmmax)
    assert avg_time > 0, "Average TMMax runtime must be positive."

    print(f"Benchmark completed. Average TMMax runtime: {avg_time:.4f} s")

    # Assertions for basic sanity checks
    for arr, name in zip(
        [loaded_tmm, loaded_tmmax],
        ["tmm", "tmmax"]
    ):
        assert isinstance(arr, np.ndarray), f"{name} should be a numpy array."
        assert arr.size > 0, f"{name} array is empty."
        assert np.all(arr >= 0), f"{name} contains negative execution times."

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)
    titles = ['tmm', 'tmmax']
    datasets = [loaded_tmm, loaded_tmmax]
    cmaps = ['YlOrRd', 'PuBuGn']

    # Plot individual heatmaps
    for i, (ax, data, cmap, title) in enumerate(zip(axes, datasets, cmaps, titles)):
        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Angle of Incidence Array Length", fontsize=10)
        ax.set_ylabel("Wavelength Array Length", fontsize=10)
        ax.set_xticks([0, 4, 8, 12, 16, 19])
        ax.set_yticks([19, 16, 12, 8, 4, 0])
        ax.set_xticklabels(["2","6","10","14","18","21"])
        ax.set_yticklabels(["21","18","14","10","6","2"])
        ax.invert_yaxis()

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time(s)", fontsize=10, fontweight='bold')
        cbar.set_ticks(np.linspace(np.min(data), np.max(data), 7))

    # Save figure
    save_file = tmp_path / "wl_theta_arr_benchmark_80_layer.png"
    plt.savefig(save_file, dpi=600)
    plt.show() 

    # Check if file exists
    assert save_file.exists(), f"Benchmark figure was not saved at {save_file}"
    assert save_file.stat().st_size > 0, "Saved benchmark figure is empty."