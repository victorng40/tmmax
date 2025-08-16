.. role:: tmmgreen :tmmgreen:`TMMax`

TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX
==========

.. image:: _static/logo_tmmax.png
   :width: 100%
   :align: center

**TMMax** is a high-performance Python library designed for the efficient simulation and analysis of optical properties in multilayer thin-film structures. Optical multilayer thin films are fundamental components in photonic systems, enabling precise control of reflectance, transmittance, and phase shift. These structures are widely used in applications such as distributed Bragg reflectors, anti-reflection coatings, spectral filters, and advanced photonic devices. Rapid and accessible simulation of these multilayer systems is therefore essential for designing and optimizing complex coatings.

TMMax addresses the limitations of traditional transfer matrix methods (TMM), which often rely on scalar calculations for each wavelength and angle of incidence, leading to redundant computations and inefficiencies in large-scale simulations. Moreover, conventional implementations generally lack support for automatic differentiation, restricting their usefulness in gradient-based inverse design and machine learning-assisted optimization.

At its core, TMMax fully vectorizes and accelerates the transfer matrix method using **JAX**, a high-performance machine learning library that enables just-in-time (JIT) compilation, vectorization, and XLA (Accelerated Linear Algebra) operations. By leveraging these capabilities, TMMax achieves exceptional computational speed, allowing users to simulate thin-film stacks with hundreds of layers in a matter of seconds. Benchmarks demonstrate that TMMax can achieve speedups of hundreds of times compared to baseline NumPy implementations, providing a significant advantage for large and complex simulations.

Originally architected for CPU-based execution to ensure accessibility across diverse hardware configurations, TMMax seamlessly extends its efficiency to **GPU** and **TPU** platforms thanks to JAX’s unified execution model. This adaptability ensures high-performance simulations across a range of computing environments without modifying the core implementation. In addition, TMMax natively supports automatic differentiation through JAX’s autograd framework, enabling the computation of analytical gradients of optical properties with respect to arbitrary system parameters. This makes TMMax particularly well-suited for **inverse design, parameter optimization, and machine learning-assisted photonic engineering**, providing a direct pathway to next-generation thin-film design.

TMMax also includes a curated **material database** and a comprehensive suite of multilayer thin-film analysis tools, giving researchers a rapid, scalable, and versatile platform for both simulation and design. Its combination of speed, accuracy, and flexibility makes TMMax a powerful Swiss Army knife for thin-film optics research.


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