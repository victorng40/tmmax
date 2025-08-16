.. role:: tmmgreen :tmmgreen:`TMMax`

TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX
==========

.. image:: _static/logo_tmmax.png
   :width: 100%
   :align: center

Optical multilayer thin films are essential building blocks in modern photonic systems, enabling precise control over reflectance, transmittance, and phase response. Fast and reliable simulation of these structures is critical for the design and analysis of advanced coatings such as distributed Bragg reflectors, anti-reflection layers, and spectral filters.

Traditional implementations of the transfer matrix method remain widely used for simulating these structures. However, their scalar treatment of wavelength and angle of incidence often leads to redundant recalculations, resulting in inefficiencies for large-scale simulations. In addition, conventional approaches lack native support for automatic differentiation, limiting their utility in gradient-based inverse design. To address these challenges, we introduce TMMax, a Python library that fully vectorizes and accelerates the transfer matrix method using the high-performance machine learning framework JAX. Designed as a versatile and extensible tool for thin-film optics research, TMMax integrates a suite of advanced numerical techniques and leverages just-in-time (JIT) compilation for optimal performance.

Originally developed with CPU-based execution to ensure broad accessibility, TMMax seamlessly scales to GPU and TPU platforms through JAX’s unified execution model. This ensures that users can benefit from both flexibility and computational efficiency, regardless of their available hardware.

TMMax also provides a curated material database with over 30 commonly used thin-film materials. Most refractive index and extinction coefficient datasets are sourced from refractiveindex.info, which compiles values from peer-reviewed literature. The database is fully extensible, and contributions from the community are encouraged. Researchers can easily add new materials by submitting issues or pull requests, supporting collaborative growth of the resource.

Beyond forward simulation, TMMax natively supports automatic differentiation via JAX’s autograd functionality. This enables analytical gradient calculations of optical properties with respect to arbitrary system parameters. Such capabilities open the door to gradient-based inverse design, optimization workflows assisted by machine learning, and parameter estimation tasks in photonics, thereby establishing TMMax as a powerful enabler for next-generation thin-film engineering.

Benchmarking demonstrates that TMMax is capable of simulating thin-film stacks containing hundreds of layers within seconds. Compared to baseline NumPy implementations, TMMax achieves speedups of several hundred times, providing a substantial advantage in computational throughput. This scalability empowers researchers to efficiently design and optimize large and complex multilayer structures, significantly accelerating the research and development cycle.

.. sidebar:: Documentation Overview

   Installation  
      Environment setup and dependencies.  

   User Guide  
      Theoretical background, motivation, and numerical benchmark results.  

   Examples  
      Demonstration cases for practical use.  

   Workshops  
      Instructional sessions designed to teach users how to work with TMMax.  

   Contributing  
      Community guidelines for participation.  

   Credits  
      References to external projects and resources that inspired this work.  

   FAQ  
      Frequently asked questions.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   examples/index
   tmmax_workshops/index
   contributing
   credits
   faq