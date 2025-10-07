"""
test_errors.py
---------------
Unit tests for input validation in the Transfer Matrix Method (TMM)
implementation of TMMax.

These tests ensure that invalid inputs raise appropriate exceptions,
confirming that the TMMax library enforces scientific consistency and
robustness across all configurations.

TMMax: High-performance modeling of multilayer thin-film structures using
the Transfer Matrix Method (TMM) with JAX.
"""

import os
import pytest
import jax.numpy as jnp
from tmmax.tmm import tmm
from tmmax.data import visualize_material_properties



# -----------------------------------------------------------------------------
# MATERIAL LIST ERRORS
# -----------------------------------------------------------------------------
def test_tmm_raises_type_error_if_material_list_not_list(
    sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise TypeError if material_list is not a list."""
    with pytest.raises(TypeError, match="material_list must be a list of material names"):
        tmm(
            material_list="SiO2",
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


def test_tmm_raises_value_error_if_material_list_empty(
    sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if material_list is empty."""
    with pytest.raises(ValueError, match="material_list cannot be empty"):
        tmm(
            material_list=[],
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


def test_tmm_raises_type_error_if_material_list_contains_non_string(
    sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise TypeError if material_list contains non-string elements."""
    with pytest.raises(TypeError, match="all elements of material_list must be strings"):
        tmm(
            material_list=["SiO2", 42, "TiO2"],
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


# -----------------------------------------------------------------------------
# THICKNESS LIST ERRORS
# -----------------------------------------------------------------------------
def test_tmm_raises_type_error_if_thickness_not_jax_array(
    sample_material_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise TypeError if thickness_list is not a JAX array."""
    with pytest.raises(TypeError, match="thickness_list must be a jax array"):
        tmm(
            material_list=sample_material_list,
            thickness_list=[100, 200, 300],
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


def test_tmm_raises_value_error_if_thickness_negative(
    sample_material_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if any thickness is negative."""
    with pytest.raises(ValueError, match="all thickness values must be positive and nonzero"):
        tmm(
            material_list=sample_material_list,
            thickness_list=jnp.array([630e-9, -200e-9, 630e-9, 200e-9, 630e-9, 200e-9]),
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


def test_tmm_raises_value_error_if_thickness_length_mismatch(
    sample_material_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if thickness_list length mismatches material_list."""
    with pytest.raises(ValueError, match="thickness_list length must match material_list length minus two"):
        tmm(
            material_list=sample_material_list,
            thickness_list=jnp.array([200e-9, 100e-9]),  # shorter list
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


# -----------------------------------------------------------------------------
# WAVELENGTH ARRAY ERRORS
# -----------------------------------------------------------------------------
def test_tmm_raises_type_error_if_wavelength_not_jax_array(
    sample_material_list, sample_thickness_list, sample_angle_of_incidences_arr
):
    """tmm() should raise TypeError if wavelength_arr is not a JAX array."""
    with pytest.raises(TypeError, match="wavelength_arr must be a jax array"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=[400e-9, 500e-9, 600e-9],
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


def test_tmm_raises_value_error_if_wavelength_nonpositive(
    sample_material_list, sample_thickness_list, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if any wavelength <= 0."""
    with pytest.raises(ValueError, match="wavelength values must be positive"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=jnp.array([500e-9, 0.0, 700e-9]),
            angle_of_incidences=sample_angle_of_incidences_arr,
        )


# -----------------------------------------------------------------------------
# ANGLE OF INCIDENCE ERRORS
# -----------------------------------------------------------------------------
def test_tmm_raises_type_error_if_angle_not_jax_array(
    sample_material_list, sample_thickness_list, sample_wavelength_arr
):
    """tmm() should raise TypeError if angle_of_incidences is not a JAX array."""
    with pytest.raises(TypeError, match="angle_of_incidences must be a jax array"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=[0, 30, 60],
        )


def test_tmm_raises_value_error_if_angle_out_of_range(
    sample_material_list, sample_thickness_list, sample_wavelength_arr
):
    """tmm() should raise ValueError if any angle < 0 or > 90 degrees."""
    with pytest.raises(ValueError, match="angles of incidence must be between 0 degree and 90 degree"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=jnp.array([-10, 1.2, 2.0]),  # radians equivalent of >90°
        )


# -----------------------------------------------------------------------------
# COHERENCY LIST ERRORS
# -----------------------------------------------------------------------------
def test_tmm_raises_type_error_if_coherency_not_list(
    sample_material_list, sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise TypeError if coherency_list is not a list."""
    with pytest.raises(TypeError, match="coherency_list must be a jnp.ndarray"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
            coherency_list=[1, 0, 1, 0, 1, 0],
        )


def test_tmm_raises_value_error_if_coherency_length_mismatch(
    sample_material_list, sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if coherency_list and material_list lengths differ."""
    with pytest.raises(ValueError, match="coherency list must have the same length as thickness list"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
            coherency_list=jnp.array([1, 0, 1]),
        )


def test_tmm_raises_value_error_if_coherency_contains_invalid_values(
    sample_material_list, sample_thickness_list, sample_wavelength_arr, sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if coherency_list contains values other than 0 or 1."""
    with pytest.raises(ValueError, match="coherency_list can only contain 0 and 1"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
            coherency_list=jnp.array([1, 0, 2, 0, -1, 0]),
        )


# -----------------------------------------------------------------------------
# POLARIZATION ERRORS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_polarization", ["x", "sp", "S", "", None])
def test_tmm_raises_value_error_if_polarization_invalid(
    invalid_polarization,
    sample_material_list,
    sample_thickness_list,
    sample_wavelength_arr,
    sample_angle_of_incidences_arr
):
    """tmm() should raise ValueError if polarization is not 's' or 'p'."""
    with pytest.raises(ValueError, match="polarization must be s or p"):
        tmm(
            material_list=sample_material_list,
            thickness_list=sample_thickness_list,
            wavelength_arr=sample_wavelength_arr,
            angle_of_incidences=sample_angle_of_incidences_arr,
            polarization=invalid_polarization,
        )