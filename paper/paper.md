---
title: 'TMMax: A high-performance JAX-based library for optical modeling of multilayer thin-film structures using the transfer matrix method'

tags:
  - Python
  - JAX
  - Optics
  - Photonics
  - Thin Film
  - Optical Coating
  - Transfer Matrix Method
  - Coating Design

authors:
  - name: Bahrem Serhat Danis
    orcid: 0009-0002-9880-0446
    equal-contrib: false
    affiliation: "1"
  - name: Esra Zayim
    orcid: 0000-0001-5887-0293
    equal-contrib: false
    affiliation: "2"


affiliations:
 - name: Department of Electrical and Electronics Engineering, Koç University, Sariyer, Istanbul, 34450, Turkey
   index: 1
 - name: Physics Engineering Department, Faculty of Science and Letters, Istanbul Technical University, Istanbul, 34469, Turkey
   index: 2

date: 15 June 2025

bibliography: paper.bib
---

# Summary

Optical multilayer thin films are fundamental components that enable the precise control of optical parameters, such as reflectance, transmittance, and phase shift, in the design of photonic systems. Rapid and accessible simulation of these structures holds critical importance for designing and analyzing complex coatings, including distributed Bragg reflectors, anti-reflection coatings, and spectral filters. While researchers commonly use the traditional transfer matrix method for designing these structures, its scalar treatment of incident light's wavelength and angle leads to redundant recalculations and inefficiencies in large-scale simulations. Furthermore, traditional method implementations do not support automatic differentiation, which limits their applicability in gradient-based inverse design approaches. Here, we present TMMax, a Python library that fully vectorizes and accelerates transfer matrix method using the high-performance machine learning library JAX. TMMax supports CPU, GPU, and TPU hardware, includes a publicly available material database, and offers comprehensive multilayer optical thin film analysis tools. Our approach, demonstrated through benchmarking, allows us to model thin film stacks with hundreds of layers within seconds. This illustrates that our method achieves a simulation speedup of x100s over a baseline NumPy implementation, providing a significant advantage in computational efficiency. Our method enables rapid and scalable simulations of large-scale and complex multilayer thin film structures, offering a substantial acceleration in the optical multilayer thin film design process. Thus, our method significantly speeds up photonics and optical engineering research.

# Statemement of need



# Code structure



# Benchmarks



# Example of usage



# Acknowledgements

The authors are grateful for feature suggestions and testing from Anahita Manchala, feedback from Liam Harnett-Caulfield, and advice from Artem Bakulin and James Durrant. This work received support from the Royal Society and the Leverhulme Trust. S.R.K. acknowledges the EPSRC Centre for Doctoral Training in the Advanced Characterisation of Materials (CDT-ACM) (EP/S023259/1) for funding a Ph.D. studentship. Via membership of the UK’s HEC Materials Chemistry Consortium, which is funded by the EPSRC (EP/L000202, EP/R029431, and EP/T022213), this work used the ARCHER2 UK National Supercomputing Service (www.archer2.ac.uk) and the UK Materials and Molecular Modelling (MMM) Hub (Young EP/T022213).

# References
