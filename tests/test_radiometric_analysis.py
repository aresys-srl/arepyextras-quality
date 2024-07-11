# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis/analysis.py core functionalities"""

import unittest

import numpy as np

from arepyextras.quality.radiometric_analysis import analysis
from arepyextras.quality.radiometric_analysis.config import ProfileExtractionParameters

random_rng1 = np.random.default_rng(12345)
raster = random_rng1.random((10, 10)) + random_rng1.random((10, 10)) * 1j


class MockParams:
    """Mocking params for these tests"""

    @property
    def filtering_kernel_size(self) -> tuple[int, int]:
        """Mocking filtering kernel size params"""
        return (5, 5)

    @property
    def outliers_kernel_size(self) -> tuple[int, int]:
        """Mocking outliers kernel size params"""
        return (5, 5)

    @property
    def smoothening_filter(self) -> bool:
        """Mocking smoothening filter flag"""
        return True

    @property
    def outliers_percentile_boundaries(self) -> tuple[float, float]:
        """Mocking outliers percentiles boundaries"""
        return [10, 90]

    @property
    def outlier_removal(self) -> bool:
        """Mocking outlier removal flag"""
        return True


class NESZProfileExtractorTest(unittest.TestCase):
    """Testing radiometric_analysis/analysis.py core nesz_profiles_extractor"""

    def setUp(self) -> None:
        self.params = ProfileExtractionParameters()
        self.tolerance = 1e-9
        self.expected_results = np.array(
            [
                -4.509810635571548,
                -3.0346688787111016,
                -1.969319798724669,
                -1.6200270090950393,
                -1.537746813328377,
                -1.6090860276809857,
                -1.806209090447819,
                -2.1745110250788366,
                -3.290663547391967,
                -4.766680877127756,
            ]
        )
        self.expected_results_default = np.array(
            [
                -5.401762464184323,
                -3.7767064179922167,
                -2.5843753462664223,
                -2.2216262106296485,
                -1.9701018004112567,
                -2.000695321727541,
                -2.2929753359972547,
                -2.716829441810087,
                -3.91720849095055,
                -5.6330122229826625,
            ]
        )

    def test_extract_nesz_profiles(self):
        """Testing NESZ profile extraction"""
        nesz_profile = analysis.nesz_profiles_extractor(data=abs(raster), params=MockParams())
        np.testing.assert_allclose(nesz_profile, self.expected_results, atol=self.tolerance, rtol=0)

    def test_extract_nesz_profiles_with_default_config(self):
        """Testing NESZ profile extraction, with default config"""
        nesz_config = analysis._nesz_config_manager(None)
        nesz_profile = analysis.nesz_profiles_extractor(
            data=abs(raster), params=nesz_config.profile_extraction_parameters
        )
        np.testing.assert_allclose(nesz_profile, self.expected_results_default, atol=self.tolerance, rtol=0)


class AverageProfileExtractorTest(unittest.TestCase):
    """Testing radiometric_analysis/analysis.py core average_elevation_profile_extractor"""

    def setUp(self) -> None:
        self.params = ProfileExtractionParameters()
        self.tolerance = 1e-9
        self.expected_results = np.array(
            [
                -6.334112329058485,
                -3.044203217967147,
                -1.77855916899467,
                -1.9295475859591928,
                -1.6781799186327797,
                -1.446758389925901,
                -1.898620508764473,
                -1.9755710045789074,
                -3.1868391218117647,
                -7.4092428227368545,
            ]
        )
        self.expected_results_default = np.array(
            [
                -np.inf,
                -9.95915910412979,
                -6.850908854900415,
                -5.157014268851174,
                -4.104151517563365,
                -4.104151517563365,
                -5.069528566107299,
                -7.108335074745231,
                -9.775345907109768,
                -np.inf,
            ]
        )

    def test_extract_gamma_profiles(self):
        """Testing Gamma profile extraction"""
        gamma_profile = analysis.average_elevation_profiles_extractor(data=abs(raster), params=MockParams())
        np.testing.assert_allclose(gamma_profile, self.expected_results, atol=self.tolerance, rtol=0)

    def test_extract_gamma_profiles_with_default_config(self):
        """Testing Gamma profile extraction, with default config"""
        gamma_config = analysis._average_elevation_config_manager(None)
        gamma_config.profile_extraction_parameters.smoothening_filter = True
        gamma_profile = analysis.average_elevation_profiles_extractor(
            data=abs(raster), params=gamma_config.profile_extraction_parameters
        )
        np.testing.assert_allclose(gamma_profile, self.expected_results_default, atol=self.tolerance, rtol=0)


class ScallopingProfileExtractorTest(unittest.TestCase):
    """Testing radiometric_analysis/analysis.py core scalloping_profiles_extractor"""

    def setUp(self) -> None:
        self.params = ProfileExtractionParameters()
        self.tolerance = 1e-9
        self.expected_results = np.array(
            [
                -6.334112329058485,
                -3.044203217967147,
                -1.77855916899467,
                -1.9295475859591928,
                -1.6781799186327797,
                -1.446758389925901,
                -1.898620508764473,
                -1.9755710045789074,
                -3.1868391218117647,
                -7.4092428227368545,
            ]
        )
        self.expected_results_default = np.array(
            [
                -0.1437943090691807,
                -0.5539229874842123,
                0.8840703135856416,
                0.7114327710913355,
                -1.1029812579587075,
                0.00945227736597075,
                -0.1313509395740905,
                -0.25745731933694127,
                -0.1288926016404768,
                0.3633151819730374,
            ]
        )

    def test_extract_scalloping_profiles(self):
        """Testing Scalloping profile extraction"""
        scalloping_profile = analysis.average_elevation_profiles_extractor(data=abs(raster), params=MockParams())
        np.testing.assert_allclose(scalloping_profile, self.expected_results, atol=self.tolerance, rtol=0)

    def test_extract_scalloping_profiles_with_default_config(self):
        """Testing Scalloping profile extraction, with default config"""
        scalloping_config = analysis._scalloping_config_manager(None)
        scalloping_profile = analysis.scalloping_profiles_extractor(
            data=abs(raster), params=scalloping_config.profile_extraction_parameters
        )
        np.testing.assert_allclose(scalloping_profile, self.expected_results_default, atol=self.tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
