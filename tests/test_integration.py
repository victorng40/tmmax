"""
Integration tests for TMMax's Transfer Matrix Method (TMM) module.

This test file ensures that the `tmm` function behaves correctly for both
coherent and incoherent multilayer stacks across polarization modes.
It validates the numerical consistency, output dimensions, and 
physical bounds of reflectance (R) and transmittance (T) results.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from tmmax.tmm import tmm

# -----------------------------------------------------------------------------
# Integration Tests (Case 1) — Fully Coherent Stack
# -----------------------------------------------------------------------------

def test_tmm_coherent_s_polarization(sample_material_list, sample_thickness_list,
                                     sample_wavelength_arr, sample_angle_of_incidences_arr):
    """Test TMM reflectance and transmittance for s-polarization in coherent mode."""
    R_s, T_s = tmm(
        material_list=sample_material_list,
        thickness_list=sample_thickness_list,
        wavelength_arr=sample_wavelength_arr,
        angle_of_incidences=sample_angle_of_incidences_arr,
        polarization='s'
    )

    # Shape and value assertions
    assert R_s.shape == (10, 100), "Reflectance (R_s) must have shape (10, 100)."
    assert T_s.shape == (10, 100), "Transmittance (T_s) must have shape (10, 100)."
    assert jnp.all((R_s >= 0) & (R_s <= 1+1e-4)), "Reflectance values must be between 0 and 1."
    assert jnp.all((T_s >= 0) & (T_s <= 1+1e-4)), "Transmittance values must be between 0 and 1."


def test_tmm_coherent_p_polarization(sample_material_list, sample_thickness_list,
                                     sample_wavelength_arr, sample_angle_of_incidences_arr):
    """Test TMM reflectance and transmittance for p-polarization in coherent mode."""
    R_p, T_p = tmm(
        material_list=sample_material_list,
        thickness_list=sample_thickness_list,
        wavelength_arr=sample_wavelength_arr,
        angle_of_incidences=sample_angle_of_incidences_arr,
        polarization='p'
    )

    # Shape and value assertions
    assert R_p.shape == (10, 100), "Reflectance (R_p) must have shape (10, 100)."
    assert T_p.shape == (10, 100), "Transmittance (T_p) must have shape (10, 100)."
    assert np.all((R_p >= 0) & (R_p <= 1+1e-4)), "Reflectance values must be between 0 and 1."
    assert np.all((T_p >= 0) & (T_p <= 1+1e-4)), "Transmittance values must be between 0 and 1."


# -----------------------------------------------------------------------------
# Integration Tests (Case 2) — Mixed Coherent/Incoherent Stack
# -----------------------------------------------------------------------------

def test_tmm_mixed_coherency_s_polarization(sample_material_list,
                                            sample_thickness_list_incoherent,
                                            sample_coherency_list,
                                            sample_wavelength_arr,
                                            sample_angle_of_incidences_arr):
    """Test TMM reflectance and transmittance for s-polarization with mixed coherency."""
    R_s, T_s = tmm(
        material_list=sample_material_list,
        thickness_list=sample_thickness_list_incoherent,
        wavelength_arr=sample_wavelength_arr,
        angle_of_incidences=sample_angle_of_incidences_arr,
        coherency_list=sample_coherency_list,
        polarization='s'
    )

    # Shape and value assertions
    assert R_s.shape == (10, 100), "Reflectance (R_s) must have shape (10, 100)."
    assert T_s.shape == (10, 100), "Transmittance (T_s) must have shape (10, 100)."
    assert np.all((R_s >= 0) & (R_s <= 1+1e-4)), "Reflectance values must be between 0 and 1."
    assert np.all((T_s >= 0) & (T_s <= 1+1e-4)), "Transmittance values must be between 0 and 1."


def test_tmm_mixed_coherency_p_polarization(sample_material_list,
                                            sample_thickness_list_incoherent,
                                            sample_coherency_list,
                                            sample_wavelength_arr,
                                            sample_angle_of_incidences_arr):
    """Test TMM reflectance and transmittance for p-polarization with mixed coherency."""
    R_p, T_p = tmm(
        material_list=sample_material_list,
        thickness_list=sample_thickness_list_incoherent,
        wavelength_arr=sample_wavelength_arr,
        angle_of_incidences=sample_angle_of_incidences_arr,
        coherency_list=sample_coherency_list,
        polarization='p'
    )

    # Shape and value assertions
    assert R_p.shape == (10, 100), "Reflectance (R_p) must have shape (10, 100)."
    assert T_p.shape == (10, 100), "Transmittance (T_p) must have shape (10, 100)."
    assert np.all((R_p >= 0) & (R_p <= 1+1e-4)), "Reflectance values must be between 0 and 1."
    assert np.all((T_p >= 0) & (T_p <= 1+1e-4)), "Transmittance values must be between 0 and 1."
