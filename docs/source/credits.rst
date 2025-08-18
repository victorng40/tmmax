.. role:: tmmgreen

Credits
=======

The development of the :tmmgreen:`TMMax` library has been influenced and supported by a number of exceptional tools, libraries, and publications in the field of optical multilayer thin films. We would like to acknowledge these contributions and provide references for further exploration.

Third-Party Libraries
--------------------

This repository occasionally utilizes approaches inspired by Steven Byrnes' [`tmm`](https://github.com/sbyrnes321/tmm) library. We sincerely thank Dr. Byrnes for developing this comprehensive and widely used resource, which has provided valuable insights for simulating multilayer optical systems.

In addition, for alternative implementations of multilayer thin-film simulations, we recommend the following libraries:

- [`vtmm`](https://github.com/fancompute/vtmm): A TensorFlow-based implementation suitable for GPU-accelerated computations.
- [`tmm-fast`](https://github.com/MLResearchAtOSRAM/tmm_fast): A PyTorch-based library with multi-threading and GPU compatibility, offering a significant performance boost for batched simulations.

These libraries provide excellent reference points and are particularly useful when GPU acceleration is available. We encourage users to explore them alongside :tmmgreen:`TMMax` to evaluate computational efficiency and performance under different conditions.

Key Publications
----------------

For calculating the optical properties of incoherent multilayer thin films, we follow the methodology introduced by:

- _Charalambos C. Katsidis and Dimitrios I. Siapkas_, "General transfer-matrix method for optical multilayer systems with coherent, partially coherent, and incoherent interference," *Applied Optics*, 41(19):3978, 2002. [doi:10.1364/AO.41.003978](https://doi.org/10.1364/AO.41.003978)

This paper provides a rigorous framework for handling coherent, partially coherent, and incoherent interference, forming a foundational reference for the calculations implemented in :tmmgreen:`TMMax`.

GPU Acceleration Considerations
-------------------------------

Certain transfer matrix method (TMM) implementations leverage GPU acceleration to enhance computational efficiency. For example:

- `vtmm` uses TensorFlow with GPU support.
- `tmm-fast` uses PyTorch with multi-threading and GPU compatibility.

As a result, benchmark comparisons may vary significantly depending on hardware availability. Users with access to GPUs are encouraged to perform their own benchmarks to compare the performance of `TMMax`, `vtmm`, and `tmm-fast` in GPU-accelerated conditions. Detailed benchmark results and implementation notes are provided in the `benchmark` directory of this repository.

Citing :tmmgreen:`TMMax`
------------------------

If you find the :tmmgreen:`TMMax` library useful in your work, we kindly ask that you consider citing it.  

The current version is available as an **arXiv** preprint. A submission to the **Journal of Open Source Software (JOSS)** is currently under review, and we will update the reference once it is published.

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