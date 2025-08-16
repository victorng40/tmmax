.. role:: tmmgreen

:tmmgreen:`TMMax`: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX
==========

.. image:: _static/logo_tmmax.png
   :width: 100%
   :align: center

Optical multilayer thin films are essential building blocks in modern photonic systems, enabling precise control over reflectance, transmittance, and phase response. **Fast** and **reliable** simulation of these structures is critical for the design and analysis of advanced coatings such as distributed Bragg reflectors, anti-reflection layers, and spectral filters.

Traditional implementations of the transfer matrix method remain widely used for simulating these structures. However, their scalar treatment of wavelength and angle of incidence often leads to redundant recalculations, resulting in inefficiencies for large-scale simulations. In addition, conventional approaches lack native support for automatic differentiation, limiting their utility in gradient-based inverse design. To address these challenges, we introduce :tmmgreen:`**TMMax**`, a Python library that fully vectorizes and accelerates the transfer matrix method using the high-performance machine learning framework `JAX <https://docs.jax.dev/en/latest/index.html>`_. Designed as a versatile and extensible tool for thin-film optics research, :tmmgreen:`TMMax` integrates a suite of advanced numerical techniques and leverages just-in-time (JIT) compilation for optimal performance.

Originally developed with CPU-based execution to ensure broad accessibility, :tmmgreen:`TMMax` seamlessly scales to GPU and TPU platforms through JAX’s unified execution model. This ensures that users can benefit from both flexibility and computational efficiency, regardless of their available hardware.

:tmmgreen:`TMMax` also provides a curated material database with over 30 commonly used thin-film materials. Most refractive index and extinction coefficient datasets are sourced from `refractiveindex.info <https://refractiveindex.info/>`_, which compiles values from peer-reviewed literature. The database is fully extensible, and contributions from the community are encouraged. Researchers can easily add new materials by submitting issues or pull requests, supporting collaborative growth of the resource.

Beyond forward simulation, :tmmgreen:`TMMax` natively supports automatic differentiation via JAX’s autograd functionality. This enables analytical gradient calculations of optical properties with respect to arbitrary system parameters. Such capabilities open the door to gradient-based inverse design, optimization workflows assisted by machine learning, and parameter estimation tasks in photonics, thereby establishing :tmmgreen:`TMMax` as a powerful enabler for next-generation thin-film engineering.

Benchmarking demonstrates that :tmmgreen:`TMMax` is capable of simulating thin-film stacks containing hundreds of layers within seconds. Compared to baseline NumPy implementations, :tmmgreen:`TMMax` achieves speedups of several hundred times, providing a substantial advantage in computational throughput. This scalability empowers researchers to efficiently design and optimize large and complex multilayer structures, significantly accelerating the research and development cycle.

.. raw:: html

   <h2>Contents</h2>

   <style>
   .tooltip {
       position: relative;
       display: inline-block;
       cursor: pointer;
       margin: 4px 0;
   }
   .tooltip .tooltiptext {
       visibility: hidden;
       width: 280px;
       background-color: #333;
       color: #fff;
       text-align: left;
       border-radius: 6px;
       padding: 6px;
       position: absolute;
       z-index: 1;
       left: 105%; /* show to the right of cursor */
       top: 50%;
       transform: translateY(-50%);
       opacity: 0;
       transition: opacity 0.3s;
   }
   .tooltip:hover .tooltiptext {
       visibility: visible;
       opacity: 1;
   }
   </style>

   <div class="tooltip"><a href="installation.html">Installation</a>
       <span class="tooltiptext">Environment setup and dependencies</span>
   </div><br>

   <div class="tooltip"><a href="quickstart.html">Quick Start</a>
       <span class="tooltiptext">First steps and essential commands</span>
   </div><br>

   <div class="tooltip"><a href="user_guide/index.html">User Guide</a>
       <span class="tooltiptext">Theoretical background, motivation, and numerical benchmark results</span>
   </div><br>

   <div class="tooltip"><a href="examples/index.html">Examples</a>
       <span class="tooltiptext">Demonstration cases for practical use</span>
   </div><br>

   <div class="tooltip"><a href="tmmax_workshops/index.html">Workshops</a>
       <span class="tooltiptext">Instructional sessions designed to teach users how to work with :tmmgreen:`TMMax`</span>
   </div><br>

   <div class="tooltip"><a href="contributing.html">Contributing</a>
       <span class="tooltiptext">Community guidelines for participation</span>
   </div><br>

   <div class="tooltip"><a href="credits.html">Credits</a>
       <span class="tooltiptext">References to external projects and resources that inspired this work</span>
   </div><br>

   <div class="tooltip"><a href="faq.html">FAQ</a>
       <span class="tooltiptext">Frequently asked questions</span>
   </div><br>

.. toctree::
   :hidden:

   installation
   quickstart
   user_guide/index
   examples/index
   tmmax_workshops/index
   contributing
   credits
   faq

Citing :tmmgreen:`TMMax`
==========

If you find the :tmmgreen:`TMMax` library useful in your work, we kindly ask that you consider citing it.  

The current version is available as an `**arXiv** preprint <https://arxiv.org/abs/2507.11341>`_. A submission to the `**Journal of Open Source Software (JOSS)** <https://github.com/openjournals/joss-reviews/issues/8686#issuecomment-3141198678>`_ is currently under review, and we will update the reference once it is published.

Here is the recommended citation in BibTeX format:

.. code-block:: latex

   @misc{danis2025tmmaxhighperformancemodelingmultilayer,
         title={TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX}, 
         author={Bahrem Serhat Danis and Esra Zayim},
         year={2025},
         eprint={2507.11341},
         archivePrefix={arXiv},
         primaryClass={physics.comp-ph},
         url={https://arxiv.org/abs/2507.11341}, 
   }
