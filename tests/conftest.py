"""
conftest.py for TMMax testing suite.

TMMax: High-performance modeling of multilayer thin-film structures
using the Transfer Matrix Method (TMM) with JAX.

This file provides:
- Shared fixtures for test setup
- Autouse setup/teardown hooks
- Reference data loaders for benchmarking
- Helper functions for assertions
"""

import pytest
import jax.numpy as jnp
import numpy as np
import os

# --------------------------------------
# Autouse fixture: setup and teardown
# --------------------------------------
@pytest.fixture(autouse=True)
def setup_environment():
    """Automatically set up and tear down the test environment."""
    print("\nSetting up test environment...")
    yield
    print("\nTearing down environment...")

# --------------------------------------
# Fixture: Sample multilayer stack
# --------------------------------------
@pytest.fixture(scope="session")
def sample_material_list():
    """Provide a reusable sample material list for reflectivity, transmission and absorption tests."""
    material_list =  ["Air", "Y2O3", "TiO2", "Y2O3", "TiO2", "Y2O3", "TiO2", "SiO2"]
    return material_list

@pytest.fixture(scope="session")
def sample_thickness_list():
    """Provide a reusable sample thickness list for reflectivity, transmission and absorption tests."""
    thickness_list = jnp.array([630e-9, 200e-9, 630e-9, 200e-9, 630e-9, 200e-9])  
    return thickness_list

@pytest.fixture(scope="session")
def sample_wavelength_arr():
    """Provide a reusable sample wavelength array for reflectivity, transmission and absorption tests."""
    wavelength_arr = jnp.linspace(500e-9, 700e-9, 100)
    return wavelength_arr

@pytest.fixture(scope="session")
def sample_angle_of_incidences_arr():
    """Provide a reusable sample angle of incidences array for reflectivity, transmission and absorption tests."""
    angle_of_incidences_arr = jnp.linspace(0, (70*jnp.pi/180), 10)
    return angle_of_incidences_arr

@pytest.fixture(scope="session")
def sample_coherency_list():
    """Provide a coherency list defining coherent/incoherent layers for integration testing."""
    return jnp.array([1, 0, 1, 0, 1, 0])


@pytest.fixture(scope="session")
def sample_thickness_list_incoherent():
    """Provide sample thicknesses for incoherent-coherent mixed stack tests."""
    return jnp.array([2000e-9, 100e-9, 2000e-9, 100e-9, 2000e-9, 100e-9])


@pytest.fixture(scope="session")
def sample_dummy_wavelengths():
    """Provide a reusable sample dummy_wavelengths for function tests."""
    dummy_wavelengths = jnp.linspace(1.5, 1.6, 100)
    return dummy_wavelengths

def dummy_material_n(wavelength_um):
    """
    Calculate refractive index for dummy material

    Args:
        wavelength_um: Wavelength in micrometers (1.5-1.6 um range)

    Returns:
        n: Refractive index
    """
    # Create a fake wavelength-dependent refractive index with some features
    n_base = 3.2
    n_peak = 0.15 * jnp.exp(-((wavelength_um - 1.55) / 0.02)**2)
    n = n_base + n_peak

    return n


def dummy_material_k(wavelength_um):
    """
    Calculate extinction coefficient for dummy material

    Args:
        wavelength_um: Wavelength in micrometers (1.5-1.6 um range)

    Returns:
        k: Extinction coefficient
    """

    # Create a fake wavelength-dependent extinction coefficient
    k_base = 0.001
    k_peak = 0.0005 * jnp.exp(-((wavelength_um - 1.53) / 0.03)**2)
    k = k_base + k_peak

    return k

@pytest.fixture(scope="session")
def sample_dummy_n():
    """Provide a reusable sample refractive index array for a function test."""
    dummy_wavelengths = jnp.linspace(1.5, 1.6, 100)
    dummy_n = dummy_material_n(dummy_wavelengths)
    return dummy_n

@pytest.fixture(scope="session")
def sample_dummy_k():
    """Provide a reusable sample extinction coefficient for a function test."""
    dummy_wavelengths = jnp.linspace(1.5, 1.6, 100)
    dummy_k = dummy_material_k(dummy_wavelengths)
    return dummy_k