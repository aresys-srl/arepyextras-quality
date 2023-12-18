# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Impulse Response analysis module"""
import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.core.masking_operations as masking
import arepyextras.quality.core.signal_processing as sp
import arepyextras.quality.point_targets_analysis.custom_dataclasses as pdt

# syncing with logger
log = logging.getLogger("quality_analysis")


class PointTargetIRFAnalysis:
    """Class performing IRF analysis for target points in a given SAR image"""

    def __init__(
        self,
        target_area: np.ndarray,
        target_pos_ref: npt.ArrayLike,
        target_pos_real: Optional[np.ndarray] = None,
        side_lobes_directions: tuple[float, float] = (np.inf, 0),
        oversampling_factor: int = 16,
        rcs_interp_factor: int = 8,
        mask_method: gdt.MaskingMethod = gdt.MaskingMethod.PEAK,
    ) -> None:
        """Init method of the class.

        Parameters
        ----------
        target_area : np.ndarray
            cropped region of the swath near the target coordinates. 2D array.
        target_pos_ref : npt.ArrayLike
            reference target position in ECEF coordinates
        target_pos_real : Optional[np.ndarray], optional
            real position of the target in the swath, if None it is evaluated, by default None
        side_lobes_directions : tuple[float, float], optional
            range and azimuth cuts angular coefficients in samples., by default (np.inf, 0)
        oversampling_factor : int, optional
            oversampling interpolation factor, by default 16
        rcs_interp_factor : int, optional
            interpolation factor for rcs evaluation, by default 8
        mask_method : gdt.MaskingMethod, optional
            masking method for interpolated peak finding, by default gdt.MaskingMethod.PEAK
        """

        # init variables
        self.target_area = target_area
        self.target_pos_ref = np.asarray(target_pos_ref)
        self.target_pos_real = target_pos_real
        self.side_lobes_directions = side_lobes_directions
        self.oversampling_factor = oversampling_factor
        self.rcs_interp_factor = rcs_interp_factor
        self.mask_method = mask_method
        self.irf_resolution_px = None

        # init functions
        self._extract_product_type()
        self._setup_variables()

    def _extract_profiles(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extracting range and azimuth profiles taking into account side lobes directions

        Parameters
        ----------
        target : np.ndarray
            input 2D array

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            range profile cut
            azimuth profile cut
        """

        # checking side_lobes_directions value, if given
        if np.isinf(self.side_lobes_directions[0]):
            rng_profile = target[:, self._roi_center_interp[1]].copy()
            az_profile = target[self._roi_center_interp[0], :].copy()

        else:
            if not np.isrealobj(target):
                rng_profile, az_profile = masking.get_interpolated_lobes_cuts(
                    x_axis=self._irf_az_axis,
                    y_axis=self._irf_rg_axis,
                    values=target,
                    side_lobes_directions=self.side_lobes_directions,
                )

        rng_profile = rng_profile / np.max(np.abs(rng_profile))
        az_profile = az_profile / np.max(np.abs(az_profile))

        return rng_profile, az_profile

    def _extract_product_type(self) -> None:
        """Checking if product image is made of purely real numbers or it is
        complex"""

        if np.isrealobj(self.target_area):
            self._data_type = gdt.TargetDataType.DETECTED
        else:
            self._data_type = gdt.TargetDataType.COMPLEX

    def _setup_variables(self) -> None:
        """Evaluating generic variables needed for the following analysis"""

        self._area_size_ratio = self.target_area.shape[1] / self.target_area.shape[0]
        self._roi = 2 * self.oversampling_factor * np.array([1, np.round(self._area_size_ratio).astype("int64")])
        self._roi_center_interp = self._roi // 2 * self.oversampling_factor

        self._irf_rg_axis = (
            np.arange(0, self._roi[0] * self.oversampling_factor) - self._roi_center_interp[0]
        ) / self.oversampling_factor
        self._irf_az_axis = (
            np.arange(0, self._roi[1] * self.oversampling_factor) - self._roi_center_interp[1]
        ) / self.oversampling_factor

        self._extract_product_type()

    @staticmethod
    def _shift_data(data: np.ndarray, center: tuple[float, float]) -> np.ndarray:
        """Shifting image in time or frequency.

        Parameters
        ----------
        data : np.ndarray
            2D array to be shifted
        center : tuple[float, float]
            center position range[0], azimuth[1]

        Returns
        -------
        np.ndarray
            shifted 2D array
        """

        # determining the shift in both range and azimuth
        rng_shift = center[0] - int(data.shape[0] // 2)
        az_shift = center[1] - int(data.shape[1] // 2)

        return sp.shift_array(data=data, row_shift=rng_shift, col_shift=az_shift)

    @staticmethod
    def _rcs_roi_extraction(
        data: np.ndarray, roi: np.ndarray, target_pos: np.ndarray = None
    ) -> tuple[int, int, np.ndarray]:
        """Extraction of a roi from the input array.

        Parameters
        ----------
        data : np.ndarray
            input array
        roi : np.ndarray
            roi_size [row number, col number]
        target_pos : np.ndarray, optional
            position of the target peak. If None, it is calculated from input array, by default None

        Returns
        -------
        tuple[int, int, np.ndarray]
            row max index
            column max index
            roi extracted from input array
        """

        if target_pos is None:
            max_row, max_col = sp.locate_max_2d(np.abs(data))
        else:
            max_row, max_col = np.floor(target_pos).astype("int64")

        # defining roi index boundaries
        row_lim_up = max_row - roi[0] // 2
        row_lim_dwn = max_row + roi[0] // 2
        col_lim_sx = max_col - roi[1] // 2
        col_lim_dx = max_col + roi[1] // 2

        # checking if roi exits array boundaries
        break_cond = np.logical_or.reduce(
            (row_lim_up < 0, row_lim_dwn > data.shape[0], col_lim_sx < 0, col_lim_dx > data.shape[1])
        )
        if break_cond:
            return max_row, max_col, None

        roi_target = data[row_lim_up:row_lim_dwn, col_lim_sx:col_lim_dx].copy()

        return max_row, max_col, roi_target

    @staticmethod
    def _rcs_peak_extraction(
        data: np.ndarray,
        target_position: Optional[np.ndarray] = None,
        max_indexes: Optional[tuple[float, float]] = None,
        interp_factor: int = 8,
    ) -> tuple[float, float]:
        """Extraction of the peak indexes from input array.

        Parameters
        ----------
        data : np.ndarray
            2D input array
        target_position : Optional[np.ndarray], optional
            position of the target peak, by default None
        max_indexes : Optional[tuple[float, float]], optional
            row and column indexes of the max. If None, it is calculated from the input array, by default None
        interp_factor : int, optional
            interpolation factor, by default 8

        Returns
        -------
        tuple[float, float]
            peak row index
            peak column index
        """

        # checking if target position is provided
        if target_position is None:
            # computing peak position
            max_row, max_col = sp.locate_max_2d(data)
        else:
            cut_row_start = max_indexes[0] * interp_factor - 1
            cut_col_start = max_indexes[0] * interp_factor - 1

            interp_int_corrected_cut = data[
                cut_row_start : cut_row_start + 2 * interp_factor + 1,
                cut_col_start : cut_col_start + 2 * interp_factor + 1,
            ]

            max_row, max_col = sp.locate_max_2d(interp_int_corrected_cut)
            max_row += int(np.floor(target_position[0]) * interp_factor) - 1
            max_col += int(np.floor(target_position[1]) * interp_factor) - 1

        return max_row, max_col

    @staticmethod
    def compute_pslr_2d(
        data: np.ndarray,
        resolution: tuple[float, float],
        interp_factor: int = 16,
        mask: gdt.MaskingMethod = gdt.MaskingMethod.RESOLUTION,
        side_lobes_directions: tuple[float, float] = np.array([np.inf, 0.0]),
    ) -> tuple[float, float, float]:
        """Compute the PSLR (Peak-to-Side-Lobe-Ratio) of the given input 2D array.

        Parameters
        ----------
        data : np.ndarray
            input 2D array to compute PSLR onto
        resolution : tuple[float, float]
            range [0] and azimuth [1] resolutions
        interp_factor : int, optional
            interpolation factor for subpixel accuracy, by default 16
        mask : gdt.MaskingMethod, optional
            masking generation method, by default gdt.MaskingMethod.RESOLUTION
        side_lobes_directions : tuple[float, float], optional
            range and azimuth cuts angular coefficients in samples, by default np.array([np.inf, 0.0])

        Returns
        -------
        tuple[float, float, float]
            range PSLR
            azimuth PSLR
            2D PSLR
        """

        # Compute data power and size
        data = np.abs(data) ** 2
        # Find data peak
        max_row, max_col = sp.locate_max_2d(data)
        main_lobe_value = data[max_row, max_col]

        # masking input data
        side_lobes_mask = masking.pslr_masking(
            data=data,
            mask_flag=mask,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
            resolution=resolution,
        )

        # masking original data
        side_lobes_data = data * side_lobes_mask
        # evaluating the max side lobes value on 2D array
        peak_row_id, peak_col_id = sp.locate_max_2d(side_lobes_data)
        max_side_lobe_value = side_lobes_data[peak_row_id, peak_col_id]

        # extracting profile cuts
        rng_cut, az_cut = masking.pslr_profile_cutting(
            masked_data=side_lobes_data,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
        )

        if np.isinf(side_lobes_directions[0]):
            max_side_lobe_value_rng = np.abs(data[np.argmax(rng_cut), max_col])
            max_side_lobe_value_az = np.abs(data[max_row, np.argmax(az_cut)])
        else:
            max_side_lobe_value_rng = np.max(rng_cut)
            max_side_lobe_value_az = np.max(az_cut)

        # Evaluate PSLR
        # PSLR 2D
        pslr_2d = sp.convert_to_db(max_side_lobe_value / main_lobe_value)

        # Azimuth PSLR
        pslr_azimuth = sp.convert_to_db(max_side_lobe_value_az / main_lobe_value)

        # Range PSLR
        pslr_range = sp.convert_to_db(max_side_lobe_value_rng / main_lobe_value)

        return pslr_range, pslr_azimuth, pslr_2d

    @staticmethod
    def compute_islr_2d(
        data: np.ndarray,
        resolution: tuple[float, float],
        interp_factor: int = 16,
        mask: gdt.MaskingMethod = gdt.MaskingMethod.RESOLUTION,
        side_lobes_directions: tuple[float, float] = np.array([np.inf, 0.0]),
    ) -> tuple[float, float, float]:
        """Compute Integral-Side-Lobe-Ratio (ISLR).

        Parameters
        ----------
        data : np.ndarray
            input 2D array to compute ISLR onto
        resolution : tuple[float, float]
            range [0] and azimuth [1] resolutions
        interp_factor : int, optional
            interpolation factor for subpixel accuracy, by default 16
        mask : gdt.MaskingMethod, optional
            masking generation method, by default gdt.MaskingMethod.RESOLUTION
        side_lobe_directions : tuple[float, float], optional
            range and azimuth cuts angular coefficients in samples, by default np.array([np.inf, 0.0])

        Returns
        -------
        tuple[float, float, float]
            range ISLR
            azimuth ISLR
            2D ISLR
        """

        # Compute data power and size
        data = np.abs(data) ** 2
        # Find data peak
        max_row, max_col = sp.locate_max_2d(data)

        main_lobe_mask, islr_mask = masking.islr_masking(
            data=data,
            mask_flag=mask,
            resolution=resolution,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
        )
        main_lobe_cuts, side_lobes_cuts = masking.islr_profile_cutting(
            data=data,
            main_lobe_mask=main_lobe_mask,
            islr_mask=islr_mask,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
        )

        # evaluating integrals over main lobe and side lobes
        main_lobe_energy = np.sum(np.abs(data * main_lobe_mask))
        side_lobes_energy = np.sum(np.abs(data * islr_mask))

        main_lobe_rng_energy = np.sum(main_lobe_cuts[0])
        main_lobe_az_energy = np.sum(main_lobe_cuts[1])
        side_lobes_rng_energy = np.sum(side_lobes_cuts[0])
        side_lobes_az_energy = np.sum(side_lobes_cuts[1])

        # Compute ISLR
        # ISLR 2D
        islr_2d = sp.convert_to_db(side_lobes_energy / main_lobe_energy)

        # Azimuth ISLR
        islr_azimuth = sp.convert_to_db(side_lobes_az_energy / main_lobe_az_energy)

        # Range ISLR
        islr_range = sp.convert_to_db(side_lobes_rng_energy / main_lobe_rng_energy)

        return islr_range, islr_azimuth, islr_2d

    @staticmethod
    def compute_sslr_2d(
        data: np.ndarray,
        resolution: tuple[float, float],
        side_lobes_directions: tuple[float, float],
        interp_factor: int = 16,
    ) -> tuple[float, float, float]:
        """Compute Secondary-Side-Lobe-Ratio (SSLR).

        Parameters
        ----------
        data : np.ndarray
            input 2D array to compute SSLR onto
        resolution : tuple[float, float]
            range [0] and azimuth [1] resolutions
        interp_factor : int, optional
            interpolation factor for subpixel accuracy, by default 16
        side_lobe_directions : tuple[float, float], optional
            range and azimuth cuts angular coefficients in samples, by default np.array([np.inf, 0.0])

        Returns
        -------
        tuple[float, float, float]
            range SSLR
            azimuth SSLR
            2D SSLR
        """

        # Compute data power and size
        data = np.abs(data) ** 2
        # Find data peak
        max_row, max_col = sp.locate_max_2d(data)
        main_lobe_value = data[max_row, max_col]

        # identifying only intermediate side lobes
        intermediate_side_lobes_mask = masking.sslr_masking(
            data=data,
            resolution=resolution,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
        )

        # evaluating the intermediate side lobes values from masked image
        masked_data = data * intermediate_side_lobes_mask
        peak_row_id, peak_col_id = sp.locate_max_2d(masked_data)
        intermediate_side_lobes = masked_data[peak_row_id, peak_col_id]

        # extracting range and azimuth cuts from masked image
        intermediate_side_lobes_values_rng, intermediate_side_lobes_values_az = masking.sslr_profile_cutting(
            masked_data=masked_data,
            peak_pos=(max_row, max_col),
            side_lobes_directions=side_lobes_directions,
            interp_factor=interp_factor,
        )

        if np.isinf(side_lobes_directions[0]):
            intermediate_side_lobes_values_rng = np.abs(data[np.argmax(intermediate_side_lobes_values_rng), max_col])
            intermediate_side_lobes_values_az = np.abs(data[max_row, np.argmax(intermediate_side_lobes_values_az)])

        # Evaluate SSLR
        # SSLR 2D
        sslr_2d = sp.convert_to_db(intermediate_side_lobes / main_lobe_value)

        # Azimuth SSLR
        sslr_azimuth = sp.convert_to_db(intermediate_side_lobes_values_az / main_lobe_value)

        # Range SSLR
        sslr_range = sp.convert_to_db(intermediate_side_lobes_values_rng / main_lobe_value)

        return sslr_range, sslr_azimuth, sslr_2d

    def compute_irf(
        self, pslr_flag: bool = True, islr_flag: bool = True, sslr_flag: bool = True, loc_errs_flag: bool = True
    ) -> tuple[pdt.IRFDataOutput, pdt.IRFGraphDataOutput]:
        """Method to compute the IRF of the input swath portion.

        Parameters
        ----------
        pslr_flag : bool, optional
            if True, performs pslr computation, by default True
        islr_flag : bool, optional
            if True, performs islr computation, by default True
        sslr_flag : bool, optional
            if True, performs sslr computation, by default True
        loc_errs_flag : bool, optional
            if True, performs localization errors computation, by default True

        Returns
        -------
        tuple[pdt.IRFDataOutput, pdt.IRFGraphDataOutput]
            dataclass object containing all computed export variables
            dataclass object containing all data needed for plotting graphs outside
        """

        # detect the position of the maximum inside the target area provided
        target = self.target_area.copy()

        # output init
        results = pdt.IRFDataOutput()
        graph_data = pdt.IRFGraphDataOutput()

        if self._data_type == gdt.TargetDataType.DETECTED:
            target = target**2

        if self.target_pos_real is None:
            # locating the peak position, if not already provided
            _, *self.target_pos_real = sp.locate_max_2d_interp(data=target)

        try:
            # re-centering target by shifting in time or frequency
            target_area_recentered = self._shift_data(data=target, center=self.target_pos_real)
        except ValueError:
            log.error("Wrong center value for target area re-centering")

            return results, graph_data

        # cropping target area around center
        target_area_recentered = sp.crop_array_2d(data=target_area_recentered, crop_size=(self._roi[0], self._roi[1]))

        # interpolating target area
        target_area_rc_interp = sp.interp2_modulated_data(
            data=target_area_recentered,
            interp_factor_az=self.oversampling_factor,
            interp_factor_rng=self.oversampling_factor,
            demod_flag_az=True,
            demod_flag_rng=True,
        )

        # extracting range and azimuth profiles taking into account lobes
        range_profile, azimuth_profile = self._extract_profiles(target=target_area_rc_interp)

        if self._data_type == gdt.TargetDataType.DETECTED:
            # root squaring the values if data is real
            range_profile = np.sqrt(range_profile)
            azimuth_profile = np.sqrt(azimuth_profile)
            target_area_rc_interp = np.sqrt(target_area_rc_interp)

        # Resolution: evaluate spatial resolution in both directions in pixels
        rng_res = sp.evaluate_irf_resolution(range_profile) / self.oversampling_factor
        az_res = sp.evaluate_irf_resolution(azimuth_profile) / self.oversampling_factor
        # storing results
        self.irf_resolution_px = (rng_res, az_res)

        if np.isnan(self.irf_resolution_px).any():
            log.error("IRF Resolution couldn't be properly assessed.")
            return results, graph_data

        results.range_resolution = self.irf_resolution_px[0]
        results.azimuth_resolution = self.irf_resolution_px[1]

        # breaking condition: something is wrong, None outputs
        if np.isnan(rng_res) or np.isnan(az_res):
            # interrupting the computation: fail
            return results, graph_data

        # PSLR, ISLR, SSLR: evaluation of specified quantities in decibel
        if pslr_flag:
            pslr = self.compute_pslr_2d(
                data=target_area_rc_interp,
                resolution=(rng_res, az_res),
                interp_factor=self.oversampling_factor,
                mask=self.mask_method,
                side_lobes_directions=self.side_lobes_directions,
            )
            # storing results
            results.range_pslr = pslr[0]
            results.azimuth_pslr = pslr[1]
            results.pslr_2d = pslr[2]
        else:
            log.info("PSLR computation has been disabled in configuration file.")
        if islr_flag:
            islr = self.compute_islr_2d(
                data=target_area_rc_interp,
                resolution=(rng_res, az_res),
                interp_factor=self.oversampling_factor,
                mask=self.mask_method,
                side_lobes_directions=self.side_lobes_directions,
            )
            # storing results
            results.range_islr = islr[0]
            results.azimuth_islr = islr[1]
            results.islr_2d = islr[2]
        else:
            log.info("ISLR computation has been disabled in configuration file.")
        if sslr_flag:
            sslr = self.compute_sslr_2d(
                data=target_area_rc_interp,
                resolution=(rng_res, az_res),
                interp_factor=self.oversampling_factor,
                side_lobes_directions=self.side_lobes_directions,
            )
            # storing results
            results.range_sslr = sslr[0]
            results.azimuth_sslr = sslr[1]
            results.sslr_2d = sslr[2]
        else:
            log.info("SSLR computation has been disabled in configuration file.")

        # Localization Error: evaluating localization error for range, azimuth
        # and ground range in pixels
        if loc_errs_flag:
            rng_loc_err = self.target_pos_real[0] - self.target_pos_ref[0]
            az_loc_err = self.target_pos_real[1] - self.target_pos_ref[1]
            ground_rng_loc_err = rng_loc_err
            # storing results
            results.slant_range_localization_error = rng_loc_err
            results.azimuth_localization_error = az_loc_err
            results.ground_range_localization_error = ground_rng_loc_err
        else:
            log.info("Localization Error computation has been disabled in configuration file.")

        # storing data for graphical output
        graph_data.image = target_area_rc_interp.copy()
        graph_data.rng_axis = self._irf_rg_axis
        graph_data.az_axis = self._irf_az_axis
        graph_data.rng_profile = range_profile.copy()
        graph_data.az_profile = azimuth_profile.copy()

        return results, graph_data

    def compute_rcs(
        self, roi_size_factor: int = 128, k_lin: int = 1, s_f: int = 1
    ) -> tuple[pdt.RCSDataOutput, pdt.RCSGraphDataOutput]:
        """Compute the Radar Cross-Section (RCS) from target acquisition data.
        Input data is considered: beta-nought, radiometrically corrected,
        absolutely calibrated (if k_lin=1) and not resampled (if s_f=1).

        Parameters
        ----------
        roi_size_factor : int, optional
            roi side size in pixel, ROI is a square, by default 128
        k_lin : int, optional
            a value of 1 means absolutely calibrated, by default 1
        s_f : int, optional
            a value of 1 means not resampled, by default 1

        Returns
        -------
        tuple[pdt.RCSDataOutput, pdt.RCSGraphDataOutput]
            dataclass object containing all computed export variables
            dataclass object containing all data needed for plotting graphs outside
        """

        target = self.target_area.copy()

        # initializing output structure
        results = pdt.RCSDataOutput()
        graphs = pdt.RCSGraphDataOutput()
        graphs.interp_factor = self.rcs_interp_factor

        if self._data_type == gdt.TargetDataType.DETECTED:
            target = target**2

        if np.isnan(self.irf_resolution_px).any():
            log.error("Could not evaluate RCS. Invalid resolution.")
            return results, graphs

        roi_size = roi_size_factor * np.array([1, np.round(self._area_size_ratio).astype("int")])
        graphs.roi_size = roi_size.copy()

        # selecting a roi centered on target area peak value
        max_row, max_col, target = self._rcs_roi_extraction(data=target, roi=roi_size, target_pos=self.target_pos_real)
        if target is None:
            log.error("Couldn't extract roi for computing RCS from target array")
            return results, graphs

        # computing intensity of target area. if data are real numbers, it has
        # already been converted into intensity before
        if self._data_type == gdt.TargetDataType.DETECTED:
            target_intensity = target.copy()
        else:
            target_intensity = np.abs(target) ** 2

        intensity_bkgnd, roi_bkgnd = sp.compute_intensity_background(
            data=target_intensity, resolutions_px=self.irf_resolution_px, roi=roi_size
        )
        results.clutter = sp.convert_to_db(intensity_bkgnd)

        # interpolate the corrected data intensity
        target_intensity_interp = sp.interp2_modulated_data(
            data=target,
            interp_factor_az=self.rcs_interp_factor,
            interp_factor_rng=self.rcs_interp_factor,
            demod_flag_az=True,
            demod_flag_rng=True,
        )

        if self._data_type == gdt.TargetDataType.DETECTED:
            target_intensity_interp = np.sqrt(target_intensity_interp)

        # correcting interpolated intensity data subtracting background
        # intensity
        target_interp_intens_corr = np.abs(target_intensity_interp) ** 2 - intensity_bkgnd

        # finding peak position in interpolated intensity corrected target area
        peak_row, peak_col = self._rcs_peak_extraction(
            data=target_interp_intens_corr,
            target_position=self.target_pos_real,
            interp_factor=self.rcs_interp_factor,
            max_indexes=(max_row, max_col),
        )
        # Peak Value: magnitude response of the target in peak position,
        # complex number
        results.peak_value_complex = target_intensity_interp[peak_row, peak_col]

        # integrate the interpolated corrected data intensity on peak region
        integrated_peak_intensity, pk_r = sp.compute_integrated_peak_intensity(
            data=target_interp_intens_corr,
            peak_position=(peak_row, peak_col),
            resolutions_px=self.irf_resolution_px,
            interp_factor=self.rcs_interp_factor,
        )

        # compute radar cross section (RCS) [per unit pixel area]
        # if results is somehow negative, 0 is returned
        results.rcs = np.max([integrated_peak_intensity / (k_lin * s_f**2), 0])
        # computing SCR
        results.scr = sp.convert_to_db(np.abs(results.peak_value_complex) ** 2) - results.clutter

        # storing values for graph plotting
        graphs.image = target
        graphs.data_type = self._data_type
        graphs.roi_background = roi_bkgnd
        graphs.roi_peak = pk_r

        return results, graphs
