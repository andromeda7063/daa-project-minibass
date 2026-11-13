# Systematic Literature Review & Methodological Foundations

This document catalogs the peer-reviewed scientific literature governing the mathematical physics, electromechanical transducer modeling, discrete optimization, and active acoustic control algorithms implemented in **MiniBass**.

---

## 1. Primary Base Papers ($\ge 2025$)

### Base Paper 1: Modal-Domain Decomposition & Placement Optimization
- **Inference Title:** Modal-Domain Decomposition & Discrete Atom Precomputation
- **Paper:** *Optimizing loudspeaker placement based on modal domain analysis for multizone sound field reproduction*, Applied Acoustics, vol. 240, p. 110904, 2025. DOI: `10.1016/j.apacoust.2025.110904`
- **Authors:** Xin Wen, Haijun Wu, Dianlong Chen, and Weikang Jiang
- **Exact Inference Used:** Decoupled cavity modal eigenfunctions from transducer electromechanics. Rather than iteratively solving complex boundary-value PDEs inside the search loop, MiniBass precomputes complex source volume velocity tensors $q_a(\omega)$ and room Green's functions $H(\mathbf{x}_r, \mathbf{x}_s, \omega)$, reducing multi-source evaluation to pure SIMD linear tensor contractions ($p = \sum H \cdot q$).

### Base Paper 2: Transducer Tolerances & Environmental Sensitivity
- **Inference Title:** Physical Tolerance Sensitivity & Environmental Perturbations
- **Paper:** *Orthogonal loudspeaker matching for stable and accurate multizone sound field reproduction*, Applied Acoustics, vol. 231, p. 110507, 2025. DOI: `10.1016/j.apacoust.2025.110507`
- **Authors:** Shang Zhao, Haoxin Ruan, Zhibin Lin, and Jing Lu
- **Exact Inference Used:** Established that multi-channel active sound reproduction systems are vulnerable to physical parameter tolerances. Operationalized in MiniBass via a 256-sample Monte Carlo sensitivity audit perturbing environmental sound speed ($\pm 2\%$), room boundary absorption $T_{60}$ ($\pm 10\%$), physical subwoofer placement ($\sigma = 5\text{ cm}$), listener head coordinates ($\sigma = 10\text{ cm}$), and voice-coil/compliance tolerances ($\pm 5\%$).

### Base Paper 3: Physical Wave Fields vs. Interpolation Artifacts
- **Inference Title:** First-Principles Wave Modeling & Interpolation Artifact Elimination
- **Paper:** *Secondary source placement optimization for sound field reproduction assisted by sound field interpolation*, Applied Acoustics, vol. 240, p. 110978, 2025. DOI: `10.1016/j.apacoust.2025.110978`
- **Authors:** Yidong Liu, Kean Chen, Lei Yang, Jianfeng Luo, and Tong Gao
- **Exact Inference Used:** Demonstrated that mathematical interpolation (e.g., nearest-neighbor or polynomial extrapolation) across discrete sensor measurements introduces severe unphysical boundary artifacts that violate the Helmholtz wave equation. MiniBass eliminates geometric interpolation entirely: all 2D sound pressure heatmaps and standing wave slices are evaluated directly from the continuous 3D Helmholtz modal Green's function across dense $40 \times 48$ spatial grids.

---

## 2. Structured 10-Paper Literature Review

1. 
- **Inference Title:** Modal-Domain Decomposition & Discrete Atom Precomputation
- **Paper:** *Optimizing loudspeaker placement based on modal domain analysis for multizone sound field reproduction*, Applied Acoustics, vol. 240, p. 110904, 2025.
- **Authors:** Xin Wen, Haijun Wu, Dianlong Chen, and Weikang Jiang
- **Exact Inference Used:** Decoupled cavity modal eigenfunctions from transducer electromechanics to precalculate complex pressure tensors, accelerating multi-source combinatorial evaluation to $>50,000$ combinations/sec.

2. 
- **Inference Title:** Physical Tolerance Sensitivity & Environmental Perturbations
- **Paper:** *Orthogonal loudspeaker matching for stable and accurate multizone sound field reproduction*, Applied Acoustics, vol. 231, p. 110507, 2025.
- **Authors:** Shang Zhao, Haoxin Ruan, Zhibin Lin, and Jing Lu
- **Exact Inference Used:** Modeled array sensitivity to component tolerances; implemented via Monte Carlo perturbation of temperature, sound speed, boundary absorption, and physical loudspeaker/seat coordinates.

3. 
- **Inference Title:** First-Principles Wave Modeling & Interpolation Artifact Elimination
- **Paper:** *Secondary source placement optimization for sound field reproduction assisted by sound field interpolation*, Applied Acoustics, vol. 240, p. 110978, 2025.
- **Authors:** Yidong Liu, Kean Chen, Lei Yang, Jianfeng Luo, and Tong Gao
- **Exact Inference Used:** Replaced numerical nearest-neighbor spatial extrapolation with first-principles 3D Helmholtz modal Green's function evaluations across dense room grids, eliminating artificial Voronoi boundaries.

4. 
- **Inference Title:** Layered Beam Search & Stepwise Candidate Pruning
- **Paper:** *Stepwise-based optimizing approaches for arrangements of loudspeaker in multi-zone sound field reproduction*, Acoustical Science and Technology, vol. 46, no. 1, pp. 11--21, 2025.
- **Authors:** Tong Zhou, Kazuya Yasueda, Ghada Bouattour, Anthimos Georgiadis, and Akitoshi Kataoka
- **Exact Inference Used:** Implemented layered discrete expansion ($d = 1, 2, 3, 4$) combined with bounded beam retention ($W=32$) and canonical position ordering, preventing combinatorial explosion while avoiding greedy trapping.

5. 
- **Inference Title:** Spatial Receiver Sampling Density & Nyquist Bounds
- **Paper:** *Effects of Reduced Information in the Performance of Low-Frequency Sound Zones*, Journal of the Audio Engineering Society, vol. 73, no. 7--8, pp. 429--445, 2025.
- **Authors:** José Cadavid, Martin Bo Møller, Toon van Waterschoot, Søren Bech, and Jan Østergaard
- **Exact Inference Used:** Evaluated sound field reconstruction under discrete sensor sampling, informing the identification of the spatial Nyquist limit ($142.9\text{ Hz}$) on the listener grid.

6. 
- **Inference Title:** Active Boundary Dipole Absorption & Wavefront Control
- **Paper:** *WaveForming: Active Acoustic Boundary Absorption for Rectangular Enclosures*, Audio Engineering Society Convention 155, Paper 10688, 2023.
- **Authors:** Arnaud Laborie, Patrice Rousseaux, and Antoine Huré
- **Exact Inference Used:** Secondary transducers configured as destructive acoustic sinks to absorb boundary reflections; synthesized in MiniBass's active stage delay and rear boundary dipole configurations to suppress longitudinal resonances.

7. 
- **Inference Title:** Multi-Seat Spatial Consistency & Seat-to-Seat Variance
- **Paper:** *Sound Reproduction: The Acoustics and Psychoacoustics of Loudspeakers and Rooms*, 3rd Edition, Focal Press / Routledge, 2018.
- **Authors:** Floyd E. Toole
- **Exact Inference Used:** Evaluated seat-to-seat spatial standard deviation across the audience footprint, establishing that global room equalization is acoustically valid only when spatial variance is suppressed.

8. 
- **Inference Title:** Geometric Modal Cancellation & Boundary Symmetry
- **Paper:** *Low-Frequency Optimization Using Multiple Subwoofers*, Journal of the Audio Engineering Society, vol. 54, no. 5, pp. 347--364, May 2006.
- **Authors:** Todd Welti and Allan Devantier
- **Exact Inference Used:** Established baseline multi-subwoofer symmetric modal cancellation; demonstrated that symmetric excitation creates shared even-order nulls, motivating MiniBass's active DSP phase and filter differentiation.

9. 
- **Inference Title:** Asymmetric Spatial Decorrelation & Modal Diversity
- **Paper:** *Multiple Subwoofers in Small Enclosures*, Audio Engineering Society Convention 121, Paper 6960, 2006.
- **Authors:** Earl R. Geddes
- **Exact Inference Used:** Intentional asymmetric physical placement decorrelates standing wave mode excitation; operationalized in MiniBass's champion layout ($x = 3.39, 4.46, 7.68, 8.75\text{ m}$) to eliminate transverse room nulls.

10. 
- **Inference Title:** Coupled Electromechanical Modeling of Vented Loudspeakers
- **Paper:** *Vented-Box Loudspeaker Systems (Parts I--IV)*, Journal of the Audio Engineering Society, vol. 21, nos. 5--8, 1973.
- **Authors:** Richard H. Small
- **Exact Inference Used:** Fourth-order coupled matrix equations mapping voice-coil voltage to diaphragm and port volume velocity, enabling physics-based simulation of vented cabinets.
