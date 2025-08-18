FAQ / Troubleshooting
=====================

Welcome to the TMMax FAQ and Troubleshooting guide. This section provides answers to common questions, issues, and guidance to help you efficiently use TMMax.

Frequently Asked Questions
--------------------------

**Q1: How do I install TMMax?**  
A1: You can install TMMax via pip:

   .. code-block:: bash

      pip install tmmax

   For a clean environment, it is recommended to create a Python virtual environment before installation. If you don’t have one, you can create it using:

   .. code-block:: bash

      python -m venv tmmax_env
      source tmmax_env/bin/activate   # On Linux/macOS
      tmmax_env\Scripts\activate      # On Windows

**Q2: What Python versions are supported?**  
A2: TMMax is compatible with Python 3.9 and above.

**Q3: How do I report a bug or request a feature?**  
A3: Please open an issue on the TMMax GitHub repository:  
   `TMMax Issues <https://github.com/bahremsd/tmmax/issues>`_

**Q4: TMMax is running slowly. How can I improve performance?**  
A4: TMMax leverages JAX for GPU/TPU acceleration. Make sure you have a GPU configured, and use JAX with GPU support. Additionally, ensure you use vectorized operations and avoid Python loops where possible.

**Q5: I encounter an error importing TMMax modules. What should I do?**  
A5: This usually occurs when dependencies are missing or your environment is misconfigured. Try reinstalling TMMax in a clean virtual environment:

   .. code-block:: bash

      pip uninstall tmmax
      pip install tmmax

   Also verify that JAX and other required packages are installed correctly.

Troubleshooting Common Issues
-----------------------------

**Issue 1: Jupyter Notebook integration doesn’t display TMMax figures**  
- Ensure you have activated the correct environment where TMMax is installed.  
- Check that `%matplotlib inline` is set at the top of your notebook.  
- Update Matplotlib and JAX to the latest versions.

**Issue 2: “ModuleNotFoundError” for JAX or TMMax**  
- Verify that your Python environment contains all dependencies: `jax`, `jaxlib`, `matplotlib`, `numpy`.  
- Activate the virtual environment where TMMax is installed before running Python or Jupyter Notebook.

**Issue 3: Errors in multilayer stack simulations**  
- Double-check your input arrays for layer thicknesses and material lists.  
- Ensure material properties are defined in your TMMax database.  
- If using custom material data, confirm the interpolation functions are correctly set.

Need More Help?
---------------

If your question is not answered here, please consult the `TMMax Documentation <https://tmmax.readthedocs.io>`_ or open a GitHub issue for assistance.

