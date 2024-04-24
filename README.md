# SAR Products Quality Analysis

`Arepyextras Quality` is the Aresys Python package for quality data processing of SAR products.

This package functionalities supports Aresys internal Product Folder format by default but they can be used providing
a protocol-compliant input object that can be generated starting from any kind of source.

The following analyses have been implemented:

- **Point Target Analysis**: Impulse Response Function (IRF), Radar Cross Section (RCS) and Localization Errors
- **Radiometric Analysis**: Noise Equivalent Sigma-Zero (NESZ), Gamma-Zero Profiles, Scalloping Profiles and custom radiometric profiles
- **Interferometric Analysis**: interferometric coherence analysis and graphical representation

The package can be installed via pip:

```shell
pip install arepyextras-quality[graphs]
```

[graphs] is an optional dependency for generating plots and graphical outputs.

or via conda:

```shell
conda install arepyextras-quality
```
