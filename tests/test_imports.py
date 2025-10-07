"""

TMMax: High-performance modeling of multilayer thin-film structures
using the Transfer Matrix Method (TMM) with JAX.


Test module for verifying that example imports in TMMax work correctly.

"""

import pytest

def test_import_visualize_material_properties():
    """Test basic imports from TMMax and matplotlib."""
    try:
        import jax.numpy as jnp
        from tmmax.data import visualize_material_properties
        import matplotlib.pyplot as plt
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_import_database_and_visualization():
    """Test imports including JAX jit and TMMax data functions."""
    try:
        import jax.numpy as jnp
        from jax import jit
        from tmmax.data import add_material_to_nk_database, visualize_material_properties
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_import_tmm_function():
    """Test import of TMMax tmm module."""
    try:
        import jax.numpy as jnp
        from tmmax.tmm import tmm
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_import_mpl_toolkits():
    """Test matplotlib and mpl_toolkits ImageGrid import."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
