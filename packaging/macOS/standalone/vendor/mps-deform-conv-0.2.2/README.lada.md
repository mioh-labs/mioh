# Lada standalone vendoring notes

- Package: `mps-deform-conv 0.2.2`
- Upstream: https://github.com/mpsops/mps-deform-conv
- Source: https://files.pythonhosted.org/packages/2b/6b/a7a0d4d90a9bde62d321a32ff383d2c288df85ccf58715062c4aa89bc2d4/mps_deform_conv-0.2.2.tar.gz
- SHA-256: `560659ba50f62f708c710a468174faccf88444fd7f5879c9390a354e054cd1d6`
- License: MIT

The standalone Python 3.12 runtime ships Torch headers that use C++20 APIs.
Lada changes only the two upstream compiler flags from `-std=c++17` to
`-std=c++20`: the PEP 517 native extension build and the runtime JIT fallback.
The Objective-C++ implementation and Metal kernel are unchanged.
