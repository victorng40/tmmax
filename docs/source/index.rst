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
       transform: translateY(-60%);
       opacity: 0;
       transition: opacity 0.3s;
   }
   .tooltip:hover .tooltiptext {
       visibility: visible;
       opacity: 1;
   }
   </style>

   <div class="tooltip"><a href="installation.html">Installation</a>
       <span class="tooltiptext">Environment setup and dependencies.</span>
   </div><br>

   <div class="tooltip"><a href="quickstart.html">Quickstart</a>
       <span class="tooltiptext">First steps and essential commands.</span>
   </div><br>

   <div class="tooltip"><a href="user_guide/index.html">User Guide</a>
       <span class="tooltiptext">Theoretical background, motivation, and numerical benchmark results.</span>
   </div><br>

   <div class="tooltip"><a href="examples/index.html">Examples</a>
       <span class="tooltiptext">Demonstration cases for practical use.</span>
   </div><br>

   <div class="tooltip"><a href="tmmax_workshops/index.html">Workshops</a>
       <span class="tooltiptext">Instructional sessions designed to teach users how to work with TMMax.</span>
   </div><br>

   <div class="tooltip"><a href="contributing.html">Contributing</a>
       <span class="tooltiptext">Community guidelines for participation.</span>
   </div><br>

   <div class="tooltip"><a href="credits.html">Credits</a>
       <span class="tooltiptext">References to external projects and resources that inspired this work.</span>
   </div><br>

   <div class="tooltip"><a href="faq.html">FAQ</a>
       <span class="tooltiptext">Frequently asked questions.</span>
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