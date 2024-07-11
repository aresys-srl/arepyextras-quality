# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/analysis.py core functionalities"""

import unittest

import numpy as np
import pandas as pd
from arepytools.io.io_support import NominalPointTarget

from arepyextras.quality.core.generic_dataclasses import (
    SARCoordinates,
    SARPolarization,
    SARSamplingFrequencies,
)
from arepyextras.quality.point_targets_analysis import analysis
from arepyextras.quality.point_targets_analysis.custom_dataclasses import (
    PointTargetAnalysisOutput,
    RCSDataOutput,
)

random_rng1 = np.random.default_rng(12345)
raster = random_rng1.random((500, 650)) + random_rng1.random((500, 650)) * 1j


class MockChannelData:
    """Mock channel to simulate a ChannelData object"""

    def read_data(self, azimuth_index: int, range_index: int, cropping_size=tuple[int, int]) -> np.ndarray:
        """Mocking the read data method.

        Parameters
        ----------
        azimuth_index : int
            slicing index along azimuth
        range_index : int
            slicing index along range
        cropping_size : _type_, optional
            cropping size along both axes, by default tuple[int, int]

        Returns
        -------
        np.ndarray
            cropped array
        """
        return raster[range_index : range_index + cropping_size[0], azimuth_index : azimuth_index + cropping_size[1]]

    @property
    def sampling_constants(self) -> SARSamplingFrequencies:
        """Mocking sampling constants"""
        return SARSamplingFrequencies(
            range_freq_hz=1, range_bandwidth_freq_hz=1, azimuth_bandwidth_freq_hz=1, azimuth_freq_hz=1
        )


class ExtractTargetAreaTest(unittest.TestCase):
    """Testing point_target_analysis/analysis.py core _extract_target_area"""

    def setUp(self) -> None:
        # creating test data
        self.channel_data = MockChannelData()
        self.sar_coordinates = SARCoordinates(azimuth=1, azimuth_index_subpx=53.22, range=1, range_index_subpx=262.17)
        self.target_area_shape = (15, 15)
        self.tolerance = 1e-9
        # expected results, case 0
        self._expected_peak_coords_case0 = np.array([7.978495751105877, 7.7960043504072445])
        self._expected_nominal_coords_case0 = np.array([13.17, 7.22])
        self._expected_peak_coords_swath_case0 = np.array([256.97849575110587, 53.796004350407244])
        self._expected_target_area_case0 = np.array(
            [
                3.377008362125439866e-01,
                1.080929074563296233e00,
                1.237778235389303205e00,
                1.070717580675390135e00,
                7.898894341815939635e-01,
                4.554044103931922938e-01,
                7.020390588626846951e-01,
                4.534360642252635576e-01,
                1.105329733801946768e00,
                9.949286809042545698e-01,
                8.685734432343359401e-01,
                9.926249371438391478e-01,
                1.079790107580713254e00,
                1.056056433036596687e00,
                1.250491290705865843e00,
                1.046006897502678390e00,
                6.698614405103572311e-01,
                8.915168660089698438e-01,
                8.187308620260537184e-01,
                1.171579273500687801e00,
                8.506389774221864153e-01,
                8.526042566929428190e-01,
                1.026056622845558097e00,
                1.261806811952206564e00,
                3.282313705068759524e-01,
                7.055470743283175361e-01,
                1.030426542464744744e00,
                6.668935430341881299e-01,
                9.189566054547396723e-01,
                2.969760586104024380e-01,
                3.709868396922019751e-01,
                7.250070736165977747e-01,
                9.961019459208821925e-01,
                1.182098578366193031e00,
                8.564120425044431384e-01,
                7.633377762184684334e-01,
                1.079370957674520071e00,
                1.005439675909622022e00,
                7.072535054466458915e-01,
                4.437242212244928430e-01,
                5.923374734574303746e-01,
                7.141511743694112235e-01,
                7.688383735108561545e-01,
                1.191365671476805943e00,
                8.460585706343132228e-01,
                3.199557734510278251e-01,
                1.047654356766370309e00,
                7.551013165351430079e-01,
                7.992680430006449122e-01,
                1.274036913595288700e00,
                7.285857967027056237e-01,
                9.842314727639371874e-01,
                1.027218055202835378e00,
                9.204284103276711981e-01,
                6.295808439701042758e-01,
                9.428319817034663819e-01,
                1.043201970187522587e00,
                9.351856916036784817e-01,
                7.687461992315705039e-01,
                5.731443835857020774e-01,
                1.100712449869725384e00,
                6.867328618934970264e-01,
                7.079341792742632755e-01,
                5.725025236163248099e-01,
                1.105912089208182403e00,
                1.003305171051544376e00,
                5.341041516444312576e-01,
                5.751505127640187398e-01,
                9.256330216837251879e-01,
                2.405915863314024294e-01,
                1.204815561538839397e00,
                6.478928031551031141e-01,
                5.229165770369211819e-01,
                6.880884001175410214e-01,
                9.345517365534742327e-01,
                9.592145611461831978e-01,
                5.570489152438204972e-01,
                4.952771194155387025e-01,
                6.106040896646900418e-01,
                9.854166896837309730e-01,
                3.993647407367267865e-01,
                9.355399274445330704e-01,
                9.729122805969101506e-01,
                8.241872462637453989e-01,
                8.113381350879752540e-01,
                1.083922964242109543e00,
                1.166535177193036787e00,
                1.005530578149356202e00,
                1.221239173294588198e00,
                8.053776027068373367e-01,
                4.232583546015291076e-01,
                7.747485921410102172e-01,
                7.997664450521168034e-01,
                9.680010473494654599e-01,
                6.693079982223801849e-01,
                9.337734585859667202e-01,
                1.344025599002589111e00,
                5.786237195469386974e-01,
                1.040655994454580346e00,
                7.703556059768641440e-01,
                6.715430003313963558e-01,
                2.760765065340962665e-01,
                8.432698534066477292e-01,
                7.352877919793977535e-01,
                1.024098475096783867e00,
                4.460703202887015273e-01,
                3.048521675722837831e-01,
                2.609969360410085981e-01,
                7.415496017728880895e-01,
                2.357908856124613095e-01,
                1.146169385634386639e00,
                1.707100446358494339e-01,
                7.147148845308443077e-01,
                7.767879425714619979e-01,
                5.916914830953203808e-01,
                3.037396157607514247e-01,
                2.736892354170439301e-01,
                9.489601830671200444e-01,
                4.218174774010314820e-01,
                1.026056542041534181e00,
                1.010395833577847791e00,
                8.998228125778744424e-01,
                9.907840183039979820e-01,
                8.091072608123851539e-01,
                1.298276966823330714e-01,
                9.469628982648120585e-01,
                5.257052093445521157e-01,
                9.198759627224564195e-01,
                4.414390053152016380e-01,
                5.002659380126502908e-01,
                1.129781747821969162e00,
                3.502320341219923860e-01,
                7.501268743550064677e-01,
                3.549837875226518613e-01,
                8.885586652764553284e-01,
                6.579300337139606381e-01,
                1.187940799450987184e00,
                4.502105323621084509e-01,
                1.005486153844073138e00,
                4.897514577116761081e-01,
                5.616154642260473562e-01,
                6.593419166169602885e-01,
                8.513649328414287787e-01,
                1.534693112659913927e-01,
                6.005742652023740025e-01,
                8.830320903298354285e-01,
                8.790080021767199270e-01,
                5.362359214610874503e-01,
                9.353643983434265508e-01,
                5.872270950229900865e-01,
                1.037304374849729038e-01,
                7.855494141532030072e-01,
                6.570321939526281296e-01,
                8.881176727268255133e-01,
                1.041277532124221583e00,
                5.602646859064138729e-01,
                3.062293479474967328e-01,
                8.761138141700036686e-01,
                1.297306696197387588e00,
                3.543152616390357612e-01,
                1.225993968427466108e00,
                7.146401468684772107e-01,
                4.780202885241773836e-01,
                8.837093131462423479e-01,
                2.989432077981255542e-01,
                7.981442967464351668e-01,
                4.776560355598599705e-01,
                1.005520950123117485e00,
                9.266812551616701032e-01,
                5.577690782519624246e-01,
                9.000007880388170012e-01,
                1.033429006624148050e00,
                7.452915899772547537e-01,
                2.678407991919627418e-01,
                1.555967560624935131e-01,
                1.040017571227330739e00,
                6.315000085756397041e-01,
                8.930218579726085792e-01,
                9.450449968821098556e-01,
                6.268203977406499039e-01,
                4.795283438566538270e-01,
                7.298865061421128120e-01,
                1.035305589425964623e00,
                6.998140170240273461e-01,
                1.027268033688085946e00,
                1.281478071800473684e00,
                9.127020896478699186e-01,
                1.192943337673208815e00,
                6.995900011314271083e-01,
                1.111695762061222670e00,
                8.846027813122991912e-01,
                6.091034658822397363e-01,
                5.579104317380951805e-01,
                1.249358972984076166e00,
                4.620503511304167032e-01,
                7.927678059841191116e-01,
                4.232192809282242507e-01,
                5.059633940307767253e-01,
                9.649780415697996050e-01,
                9.935437046377707526e-01,
                6.878755975843542370e-01,
                5.671319609866547484e-01,
                6.194529750536158508e-01,
                4.057587495684672474e-01,
                9.740675307436106323e-01,
                6.805115005149171559e-01,
                7.223347924881283744e-01,
                7.653217204985452993e-01,
                9.538710729384503040e-01,
                1.023348376437789975e00,
                6.246224441735367394e-01,
                7.356554458888701076e-01,
                8.710278688978555506e-01,
                8.517046609634727483e-01,
                9.905070513020489686e-01,
                7.366398981352932918e-01,
                9.672849642754680621e-01,
                4.404311734241500575e-01,
                7.407583681282197130e-01,
                5.127912514667445354e-01,
                4.961286664542802205e-01,
                8.511932846503438288e-01,
                8.495355664475889856e-01,
                9.135973788081531710e-01,
                1.084325222720803339e00,
            ]
        ).reshape(self.target_area_shape)
        # expected results, case 1
        self._expected_peak_coords_case1 = np.array([7.573835804778412, 7.871132428395738])
        self._expected_nominal_coords_case1 = np.array([5.17, 6.22])
        self._expected_peak_coords_swath_case1 = np.array([264.5738358047784, 54.87113242839574])
        self._expected_target_area_case1 = np.array(
            [
                8.998228125778744424e-01,
                9.907840183039979820e-01,
                8.091072608123851539e-01,
                1.298276966823330714e-01,
                9.469628982648120585e-01,
                5.257052093445521157e-01,
                9.198759627224564195e-01,
                4.414390053152016380e-01,
                5.002659380126502908e-01,
                1.129781747821969162e00,
                3.502320341219923860e-01,
                7.501268743550064677e-01,
                3.549837875226518613e-01,
                8.885586652764553284e-01,
                3.700107594191375804e-01,
                1.187940799450987184e00,
                4.502105323621084509e-01,
                1.005486153844073138e00,
                4.897514577116761081e-01,
                5.616154642260473562e-01,
                6.593419166169602885e-01,
                8.513649328414287787e-01,
                1.534693112659913927e-01,
                6.005742652023740025e-01,
                8.830320903298354285e-01,
                8.790080021767199270e-01,
                5.362359214610874503e-01,
                9.353643983434265508e-01,
                5.872270950229900865e-01,
                5.637564336664239173e-01,
                7.855494141532030072e-01,
                6.570321939526281296e-01,
                8.881176727268255133e-01,
                1.041277532124221583e00,
                5.602646859064138729e-01,
                3.062293479474967328e-01,
                8.761138141700036686e-01,
                1.297306696197387588e00,
                3.543152616390357612e-01,
                1.225993968427466108e00,
                7.146401468684772107e-01,
                4.780202885241773836e-01,
                8.837093131462423479e-01,
                2.989432077981255542e-01,
                9.435321209004897680e-01,
                4.776560355598599705e-01,
                1.005520950123117485e00,
                9.266812551616701032e-01,
                5.577690782519624246e-01,
                9.000007880388170012e-01,
                1.033429006624148050e00,
                7.452915899772547537e-01,
                2.678407991919627418e-01,
                1.555967560624935131e-01,
                1.040017571227330739e00,
                6.315000085756397041e-01,
                8.930218579726085792e-01,
                9.450449968821098556e-01,
                6.268203977406499039e-01,
                8.282046463518418777e-01,
                7.298865061421128120e-01,
                1.035305589425964623e00,
                6.998140170240273461e-01,
                1.027268033688085946e00,
                1.281478071800473684e00,
                9.127020896478699186e-01,
                1.192943337673208815e00,
                6.995900011314271083e-01,
                1.111695762061222670e00,
                8.846027813122991912e-01,
                6.091034658822397363e-01,
                5.579104317380951805e-01,
                1.249358972984076166e00,
                4.620503511304167032e-01,
                8.899411494641307208e-01,
                4.232192809282242507e-01,
                5.059633940307767253e-01,
                9.649780415697996050e-01,
                9.935437046377707526e-01,
                6.878755975843542370e-01,
                5.671319609866547484e-01,
                6.194529750536158508e-01,
                4.057587495684672474e-01,
                9.740675307436106323e-01,
                6.805115005149171559e-01,
                7.223347924881283744e-01,
                7.653217204985452993e-01,
                9.538710729384503040e-01,
                1.023348376437789975e00,
                7.563303538375981683e-01,
                7.356554458888701076e-01,
                8.710278688978555506e-01,
                8.517046609634727483e-01,
                9.905070513020489686e-01,
                7.366398981352932918e-01,
                9.672849642754680621e-01,
                4.404311734241500575e-01,
                7.407583681282197130e-01,
                5.127912514667445354e-01,
                4.961286664542802205e-01,
                8.511932846503438288e-01,
                8.495355664475889856e-01,
                9.135973788081531710e-01,
                1.084325222720803339e00,
                4.922770373228523288e-01,
                3.702514761391846876e-01,
                1.273830931786807508e00,
                7.245511140684044449e-01,
                4.142603556051412950e-01,
                1.162424923694962864e00,
                7.677503540763080947e-02,
                9.870391861976920422e-01,
                7.866290116070798755e-01,
                2.701488422707288839e-01,
                6.502412145582792125e-01,
                1.588569622595877584e-01,
                7.586750576371646959e-01,
                6.809685628338125740e-01,
                8.306653917596752423e-01,
                9.573545157056960742e-01,
                1.000852657938709367e00,
                1.001411750566685965e00,
                4.325621848038866957e-01,
                7.246848070153493193e-01,
                7.090799588407428189e-01,
                9.524484786944069636e-01,
                7.330791437815851186e-01,
                6.692620570077845166e-01,
                5.836197626542372063e-01,
                3.508091552537970781e-01,
                6.831792398195228877e-01,
                1.209254210648092354e00,
                1.194736764155463593e00,
                6.181699603072057236e-01,
                1.174497926473713871e00,
                1.060828455140491622e00,
                9.092402064402745898e-01,
                1.161646492611948966e00,
                1.182811269713944569e00,
                8.855757894349448600e-01,
                1.168047337041324241e00,
                5.508251389926800412e-01,
                3.402937370126604133e-01,
                5.925560459230537758e-01,
                9.407810962085009221e-01,
                2.131014746835485640e-01,
                7.801027409926529765e-01,
                5.923943520435082100e-01,
                6.428549356353868705e-01,
                3.469403169061233139e-01,
                5.279111334580982584e-01,
                4.393082000026234946e-01,
                7.134808345682922548e-01,
                4.611353983086732278e-01,
                6.315410728486047232e-01,
                4.024163071380931211e-01,
                7.828360597519458119e-01,
                1.086718648152282585e00,
                6.250618448446290110e-01,
                1.174052490380770974e00,
                9.189112520218876146e-01,
                4.431322401485491835e-01,
                9.586469042783037908e-01,
                5.812454083661990900e-01,
                3.329932456576774613e-01,
                1.260878664260218551e00,
                4.928045621155661138e-01,
                1.015068096292071953e00,
                6.822635963496342937e-01,
                1.093900669965237604e00,
                7.327501996823668762e-01,
                7.707029788403568782e-01,
                7.702829608922638904e-01,
                1.011680847184223131e00,
                7.953858260389635726e-01,
                9.664515593067423138e-01,
                7.241255214841819665e-01,
                6.173764606014162659e-01,
                6.170442881477835595e-01,
                7.385624082112675071e-01,
                6.238476211853523079e-01,
                3.957204680032321287e-01,
                7.193809778621798046e-01,
                1.227483480357241863e00,
                9.313901723953580403e-01,
                6.154795657704320577e-01,
                1.154706081812421958e00,
                8.254779023376201952e-01,
                1.167872410767790470e00,
                7.162308750810207902e-01,
                9.909341186193405537e-01,
                2.877123817347020940e-01,
                1.102775453026079600e00,
                9.599130812620906283e-01,
                9.532056319762470808e-01,
                9.358316510772716201e-01,
                7.657095134778517664e-01,
                7.339735924019576974e-01,
                1.383442943836225292e-01,
                7.161111546807084061e-01,
                3.102107317629875238e-01,
                9.053662380154936518e-01,
                6.662227532334341618e-01,
                8.097155961178629235e-01,
                8.645159430407991508e-01,
                1.075749267165771794e00,
                1.045320651444794496e-01,
                9.344023225384016085e-01,
                5.138593235443343898e-01,
                5.656186644994313228e-01,
                6.709180950144223443e-01,
                1.101073429451266072e00,
                5.944217139406157147e-01,
                6.576405688526709703e-01,
                8.253556920704256950e-01,
                4.442497298983518750e-01,
                2.552299195826454037e-01,
                8.378712200227618201e-01,
                1.066609568087080184e00,
                1.013016331320236718e00,
                7.912620618428201080e-01,
                1.017795935897780168e00,
                8.480666072241207498e-01,
                9.373046170394430110e-01,
                2.529486751221610152e-01,
            ]
        ).reshape(self.target_area_shape)

    def test_extract_target_area_case0(self) -> None:
        """Testing _extract_target_area function, no crops info"""
        target_area, peak_coords, nominal_coords, peak_coords_swath = analysis._extract_target_area(
            channel_data=self.channel_data,
            azimuth_range_coordinates=self.sar_coordinates,
            initial_crop=(10, 10),
            final_crop=self.target_area_shape,
        )
        self.assertIsInstance(target_area, np.ndarray)
        self.assertIsInstance(peak_coords, np.ndarray)
        self.assertIsInstance(nominal_coords, np.ndarray)
        self.assertIsInstance(peak_coords_swath, np.ndarray)
        self.assertEqual(target_area.shape, self.target_area_shape)
        self.assertEqual(peak_coords.shape, (2,))
        self.assertEqual(nominal_coords.shape, (2,))
        self.assertEqual(peak_coords_swath.shape, (2,))

        np.testing.assert_allclose(np.abs(target_area), self._expected_target_area_case0, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(peak_coords, self._expected_peak_coords_case0, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(nominal_coords, self._expected_nominal_coords_case0, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(
            peak_coords_swath, self._expected_peak_coords_swath_case0, atol=self.tolerance, rtol=0
        )

    def test_extract_target_area_case2(self) -> None:
        """Testing _extract_target_area function, with ale limits"""
        target_area, peak_coords, nominal_coords, peak_coords_swath = analysis._extract_target_area(
            channel_data=self.channel_data,
            azimuth_range_coordinates=self.sar_coordinates,
            ale_limits=(16, 8),
            final_crop=self.target_area_shape,
        )
        self.assertIsInstance(target_area, np.ndarray)
        self.assertIsInstance(peak_coords, np.ndarray)
        self.assertIsInstance(nominal_coords, np.ndarray)
        self.assertIsInstance(peak_coords_swath, np.ndarray)
        self.assertEqual(target_area.shape, self.target_area_shape)
        self.assertEqual(peak_coords.shape, (2,))
        self.assertEqual(nominal_coords.shape, (2,))
        self.assertEqual(peak_coords_swath.shape, (2,))

        np.testing.assert_allclose(np.abs(target_area), self._expected_target_area_case1, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(peak_coords, self._expected_peak_coords_case1, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(nominal_coords, self._expected_nominal_coords_case1, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(
            peak_coords_swath, self._expected_peak_coords_swath_case1, atol=self.tolerance, rtol=0
        )


class AddUnitOfMeasureTest(unittest.TestCase):
    """Testing point_target_analysis/analysis.py core _add_unit_of_measure"""

    def setUp(self) -> None:
        # creating test data
        self.columns = [
            "azimuth_resolution",
            "squint_angle",
            "pslr_2d",
            "rcs_error",
            "ground_velocity",
            "incidence_angle",
            "islr_2d",
        ]
        self.df = pd.DataFrame(np.random.randint(0, 10, size=(10, 7)), columns=self.columns)
        self.expected_columns = [
            "azimuth_resolution_[m]",
            "squint_angle_[rad]",
            "pslr_2d_[dB]",
            "rcs_error_[dB]",
            "ground_velocity_[ms]",
            "incidence_angle_[deg]",
            "islr_2d_[dB]",
        ]

    def test_add_unit_of_measure(self) -> None:
        """Testing _add_unit_of_measure function"""
        cols = analysis._add_unit_of_measure(self.df.columns)
        self.assertListEqual(cols, self.expected_columns)


class ResultsToDataframeTest(unittest.TestCase):
    """Testing point_target_analysis/analysis.py core _results_to_dataframe"""

    def setUp(self) -> None:
        # creating test data
        self.data = [PointTargetAnalysisOutput()]
        # expected results
        self.expected_values = np.zeros((1, 39)) + np.nan

    def test_results_to_dataframe(self) -> None:
        """Testing _results_to_dataframe function"""
        df = analysis._results_to_dataframe(self.data)
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_array_equal(df.values.astype(float), self.expected_values)


class ComputeAdditionalRCSValuesTest(unittest.TestCase):
    """Testing point_target_analysis/analysis.py core _compute_additional_rcs_values"""

    def setUp(self) -> None:
        # creating test data
        self.rcs_data = RCSDataOutput(
            clutter=10,
            peak_phase_error=126,
            peak_value_complex=124 + 3.6j,
            rcs=122.54,
            rcs_error=108.3,
            scr=8,
        )
        self.rng_step = 1.34
        self.az_step = 1.22
        self.interp = 7
        self.pol = SARPolarization.VH
        self.sat_pos = np.array([7580000, 1520000, 1250253])
        self.nominal_target = NominalPointTarget(
            xyz_coordinates=np.array([7557000, 1480000, 1130783]), rcs_vh=2000 + 0j
        )
        self.carrier_freq = 900000000
        self.tolerance = 1e-9
        # expected results
        self.expected_rcs = (4.088334530612245, 6.115464250069774, -26.89483570657004, -134.10414221664118)

    def test_results_to_dataframe(self) -> None:
        """Testing _compute_additional_rcs_values function"""
        rcs_values = analysis._compute_additional_rcs_values(
            rcs_input=self.rcs_data,
            step_distances=(self.rng_step, self.az_step),
            interp_factor=self.interp,
            polarization=self.pol,
            target_info=self.nominal_target,
            sat_position=self.sat_pos,
            fc_hz=self.carrier_freq,
        )
        self.assertIsInstance(rcs_values, tuple)
        np.testing.assert_allclose(np.array(rcs_values), np.array(self.expected_rcs), atol=self.tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
