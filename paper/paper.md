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
 - name: Physics Engineering Department, Istanbul Technical University, Istanbul, 34469, Turkey
   index: 2

date: 16 July 2025

bibliography: paper.bib
---

# Summary

Optical multilayer thin films are fundamental components that enable the precise control of optical parameters, such as reflectance, transmittance, and phase shift, in the design of photonic systems. Rapid and accessible simulation of these structures holds critical importance for designing and analyzing complex coatings, including distributed Bragg reflectors, anti-reflection coatings, and spectral filters. While researchers commonly use the traditional transfer matrix method for designing these structures, its scalar treatment of incident light's wavelength and angle leads to redundant recalculations and inefficiencies in large-scale simulations. Furthermore, traditional method implementations do not support automatic differentiation, which limits their applicability in gradient-based inverse design approaches. Here, we present TMMax, a Python library that fully vectorizes and accelerates transfer matrix method using the high-performance machine learning library JAX. TMMax supports CPU, GPU, and TPU hardware, includes a publicly available material database, and offers comprehensive multilayer optical thin film analysis tools. Our approach, demonstrated through benchmarking, allows us to model thin film stacks with hundreds of layers within seconds. This illustrates that our method achieves a simulation speedup of x100s over a baseline NumPy implementation, providing a significant advantage in computational efficiency. Our method enables rapid and scalable simulations of large-scale and complex multilayer thin film structures, offering a substantial acceleration in the optical multilayer thin film design process. Thus, our method significantly speeds up photonics and optical engineering research.

# Statemement of need

$$
\mathbf{M} = \mathbf{I}_0 \cdot \left( \prod_{i=1}^{N-2} \mathbf{M}_i \right)
\tag{1}
\newline
\mathbf{M}_i = \mathbf{I}_i \mathbf{P}_i = 
\begin{bmatrix}
\alpha_{i, i+1} & \gamma_{i, i+1} \\
\gamma_{i, i+1} & \alpha_{i, i+1}
\end{bmatrix}
\begin{bmatrix}
e^{-j \delta_i} & 0 \\
0 & e^{j \delta_i}
\end{bmatrix}
=
\begin{bmatrix}
\alpha_{i, i+1} e^{-j \delta_i} & \gamma_{i, i+1} e^{j \delta_i} \\
\gamma_{i, i+1} e^{-j \delta_i} & \alpha_{i, i+1} e^{j \delta_i}
\end{bmatrix}
$$

$$
\alpha_{i, i+1} =
\begin{cases}
\dfrac{n_i \cos\theta_i + n_{i+1} \cos\theta_{i+1}}{2 n_i \cos \theta_i} & \text{(s-pol.)} \\
\dfrac{n_i \cos\theta_{i+1} + n_{i+1} \cos\theta_{i}}{2 n_i \cos \theta_i} & \text{(p-pol.)}
\end{cases}, \quad
\gamma_{i, i+1} =
\begin{cases}
\dfrac{n_i \cos\theta_i - n_{i+1} \cos\theta_{i+1}}{2 n_i \cos \theta_i} & \text{(s-pol.)} \\
\dfrac{n_i \cos\theta_{i+1} - n_{i+1} \cos\theta_{i}}{2 n_i \cos \theta_i} & \text{(p-pol.)}
\end{cases}
\tag{2}
$$

$$
\delta_i = \dfrac{2\pi}{\lambda} n_i d_i \cos \theta_i
\tag{3}
$$


\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figure1.pdf}
\caption{Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.}
\end{figure*}


# Code structure

\begin{figure*}[t]
\centering\includegraphics[width=0.92\textwidth]{figure2.pdf}
\caption{Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.}
\end{figure*}


# Benchmarks

\begin{figure*}[t]
\centering\includegraphics[width=0.92\textwidth]{figure3.pdf}
\caption{Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.}
\end{figure*}

\begin{figure*}[t]
\centering\includegraphics[width=0.92\textwidth]{figure4.pdf}
\caption{Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.}
\end{figure*}


# Acknowledgements

This work was supported by the Scientific and Technological Research Council of Türkiye (TUBITAK) under the 2209-A Research Project Support Programme for Undergraduate Students, 2022 First-Term Call.

# References
