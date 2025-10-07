import os
import pytest
import jax.numpy as jnp
from tmmax.data import visualize_material_properties, add_material_to_nk_database

# ---------- TEST FOR visualize_material_properties ----------

@pytest.mark.parametrize("logX, logY", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_visualize_material_properties_creates_file(tmp_path, logX, logY):
    """
    Test visualize_material_properties function with different log scales.
    Checks if the PNG file is correctly created in the specified folder.
    """
    folder_path = tmp_path / f"SiO2_logX_{logX}_logY_{logY}"
    folder_path.mkdir()
    
    # Call the function
    visualize_material_properties(
        material_name="SiO2",
        logX=logX,
        logY=logY,
        eV=True,
        savefig=True,
        save_path=str(folder_path)
    )
    
    # Assert PNG file exists
    expected_file = folder_path / "SiO2_nk_plot.png"
    assert expected_file.exists(), f"Expected file {expected_file} does not exist."

# ---------- TEST FOR add_material_to_nk_database ----------

def test_add_material_to_nk_database_creates_csv(sample_dummy_wavelengths, sample_dummy_n, sample_dummy_k):
    """
    Test add_material_to_nk_database to verify that a new CSV file is created
    for the dummy material inside tmmax/nk_data/csv.
    """
    material_name = "dummy_material"
    base_path = os.path.join("tmmax", "nk_data", "csv")
    csv_path = os.path.join(base_path, f"{material_name}.csv")

    # Remove the file if it already exists from a previous run
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Call the function
    add_material_to_nk_database(
        wavelength_arr=sample_dummy_wavelengths,
        refractive_index_arr=sample_dummy_n,
        extinction_coeff_arr=sample_dummy_k,
        material_name=material_name,
    )

    # Check if the file was created successfully
    assert os.path.exists(csv_path), f"Expected CSV file was not created at {csv_path}"
