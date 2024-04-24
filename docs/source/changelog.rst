Changelog
=========

v1.1.1
------

**Other changes**

- `Point Target Analysis`: incidence angle is now computed at target location and not at mid-range
- `Point Target Analysis`: RCS error computation improved and simplified
- `Point Target Analysis`: sign of Azimuth and Range absolute localization errors has been changed (ESA convention)
- `Point Target Analysis`: added `peak_azimuth_from_burst_start` output column
- `Interferometric analysis` module added: coherence computation from interferogram and coherence 2D histogram computation along range and azimuth
- `Interferometric analysis` output: azimuth and range coherence 2D histograms dump to NetCDF, graphs module added
- `Radiometric Analysis`: added a block-wise radiometric analysis feature grouping several radiometric profiles implementations,
  in particular Noise Equivalent Sigma-Zero (NESZ), Gamma-Zero (:math:`\gamma^0`) Profiles and Scalloping Profiles
- `Radiometric Analysis` output: profiles and 2D histogram saving to NetCDF, graphs module added
- A generic function ``radiometric_profiles`` can be used to fully customize profiles extraction from a SAR product

**Incompatible changes**

- Point-wise Radiometric Analysis removed from this package, moved to `arepyextras-sqt`
- CLI tool no more available, moved to `arepyextras-sqt`
- Configuration from toml has been removed, it's related only to the CLI tool and therefore moved to `arepyextras-sqt`
- Configuration module removed
- Point target analysis and radiometric analysis dataclass configurations moved from their respective `custom_dataclasses` files to new `config` files
- Removed previous `NESZ Analysis` implementation
- ``convert_to_db`` function updated and ``DecibelConversion`` enum edited

**Other changes**

- Improved automatic generation of custom extraction ROI based on externally provided ALE limits in `point_target_analysis`
- `quality_input_protocol` minor improvements
- Documentation updated

**Bug Fixing**

- Minor bug fixed in `interferometric_analysis.config` module when loading from dict
- Minor bug fixed in `interferometric_analysis.support` module when computing coherence and masking invalid data
- Bug fixed in `point_target_analysis` module related to impossibility to find maximum with `locate_max_2d_interp` in a target area full of zeroes
- Bug fixed in `point_target_analysis` module related to ROI extraction for asymmetric ROIs
- Minor other fixes

v1.0.0
------

**New Features**

- Introduced `Arepytools > 1.5.0` as a requirement to use at best the new geocoding functionalities
- `core.signal_processing.py` documentation highly improved and minor refactoring
- Topsar (SLC) product supported for both point target analysis and radiometric analysis
- Squinted data products supported for point target analysis

**Other changes**

- Integration tests added in CI to keep controlled the whole point target analysis functionality
- Removed `core._additional_tools` private module
- Documentation updated
