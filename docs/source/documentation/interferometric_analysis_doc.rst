.. _quality_inter:

Interferometric Analysis
========================

This analysis consist in computing 2D coherence intensity histograms along range and azimuth directions from SAR products
containing interferogram information or pre-computed coherence values.

Computed and Estimated quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coherence is computed as the ratio between an interferogram's complex data values and their magnitude after applying a Boxcar
filter with a specific kernel to the whole image. Image is processed burst by burst so to keep the information as relevant as
possible.

The results of this computation are two 2D coherence intensity histograms along both SAR dimensions computed after partitioning
each burst data for a configurable number of times along each direction.

Analysis Algorithm
^^^^^^^^^^^^^^^^^^

The interferometric analysis algorithm is divided into two main stages, depending on the input product nature.
If interferogram data are provided, coherence must be computed (by setting to ``True`` the configuration variable
`enable_coherence_computation`) before evaluating the 2D intensity histograms. Otherwise, if the input product already
contains coherence values, this last operation can be directly performed.


Coherence Computation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each channel of the input product is separately analyzed and also if the product is not merged, i.e. it's still subdivided
into bursts, this feature is preserved and data are processed burst by burst.

For each data portion to be processed, coherence is computed by performing a 2D convolution with a normalized rectangular
Boxcar filter both on input complex data and on its magnitude and evaluating the ratio between the two.

2D Coherence Intensity Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2D histograms to determine the distribution of coherence intensity in the scene are computed for each data portion after
partitioning it in blocks along a given direction and determining the histogram on each of those blocks. This operation
is performed along both SAR directions (range and azimuth) and saved as an NxM array with shape equal to the number of 
the number of coherence intensity bins and the number of partitioning block along that direction.


Analysis Output
^^^^^^^^^^^^^^^

Interferometric analysis output consists in a .nc NetCDF4 file containing the 2D coherence histograms computed along both
directions. Also, a graphical plot can be obtained as output using the ``graphical_output.radiometric_2D_hist_plot``
functionality showing the coherence map and the two histograms along the corresponding axes.

.. note::

    Graphical output functionalities are available only if the package has been installed with the [graphs] optional
    dependencies. Refer to the :ref:`installation documentation<quality_install>` for more information.

