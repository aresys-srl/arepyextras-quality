# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for interferometric_analysis/graphical_output.py core functionalities"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from arepyextras.quality.core.generic_dataclasses import SARPolarization
from arepyextras.quality.interferometric_analysis.custom_dataclasses import (
    InterferometricCoherence2DHistograms,
    InterferometricCoherenceOutput,
)
from arepyextras.quality.interferometric_analysis.graphical_output import (
    generate_coherence_graphs,
)


@unittest.skipIf(sys.platform.startswith("win"), "skipping Windows on CI")
class InterferometricAnalysisGraphicalOutputTest(unittest.TestCase):
    """Testing Interferometric Analysis graphical output functionalities"""

    def setUp(self) -> None:
        # test data
        self._coherence = np.array(
            [
                0.6087167 + 0.66993212j,
                0.64719964 + 0.62887204j,
                0.63121965 + 0.62820172j,
                0.6184499 + 0.63734102j,
                0.62694573 + 0.63785247j,
                0.66934961 + 0.60926945j,
                0.644497 + 0.65384073j,
                0.65316248 + 0.65211705j,
                0.67542878 + 0.64360748j,
                0.65586474 + 0.66881043j,
                0.63200926 + 0.70168228j,
                0.63681054 + 0.71331348j,
                0.65612008 + 0.69060612j,
                0.65551559 + 0.68505891j,
                0.61925352 + 0.70148133j,
                0.60764265 + 0.6994434j,
                0.60648372 + 0.69637524j,
                0.61000102 + 0.69142071j,
                0.64972981 + 0.65700794j,
                0.66146537 + 0.63376793j,
                0.61427607 + 0.64940212j,
                0.65754156 + 0.61068425j,
                0.63876016 + 0.62277021j,
                0.62090749 + 0.63709718j,
                0.63153756 + 0.63330798j,
                0.65853444 + 0.62218853j,
                0.6585467 + 0.64953378j,
                0.6560291 + 0.65770369j,
                0.66421708 + 0.6605654j,
                0.64856272 + 0.67932274j,
                0.630873 + 0.70528317j,
                0.62915053 + 0.71847274j,
                0.63760664 + 0.70801597j,
                0.6375095 + 0.70352472j,
                0.61509266 + 0.70770651j,
                0.62024971 + 0.69485375j,
                0.60807845 + 0.69604186j,
                0.60191478 + 0.70185237j,
                0.63108965 + 0.67257044j,
                0.65472846 + 0.64281284j,
                0.60843657 + 0.67761494j,
                0.65611091 + 0.63512682j,
                0.62970838 + 0.65164193j,
                0.6087502 + 0.66553431j,
                0.62078817 + 0.65891678j,
                0.63415332 + 0.65507088j,
                0.63888396 + 0.67614964j,
                0.64226997 + 0.67683361j,
                0.65441544 + 0.67500436j,
                0.63733008 + 0.69441672j,
                0.62834949 + 0.71311801j,
                0.63931307 + 0.71148091j,
                0.63437599 + 0.70876914j,
                0.64137577 + 0.7012274j,
                0.62728666 + 0.69759712j,
                0.62686964 + 0.69107377j,
                0.61441737 + 0.69424908j,
                0.60740554 + 0.69957788j,
                0.6339779 + 0.67061599j,
                0.64346963 + 0.6587807j,
                0.62847336 + 0.65412599j,
                0.66977578 + 0.62250678j,
                0.64138052 + 0.64374477j,
                0.61879873 + 0.65976941j,
                0.62332984 + 0.65929181j,
                0.63295855 + 0.6577987j,
                0.63846701 + 0.67623824j,
                0.63266148 + 0.69043521j,
                0.63227017 + 0.69825089j,
                0.62211213 + 0.70407693j,
                0.61952944 + 0.71446683j,
                0.63525664 + 0.70853697j,
                0.62777932 + 0.70729893j,
                0.63790986 + 0.69840102j,
                0.63183035 + 0.68810788j,
                0.63413436 + 0.6812227j,
                0.62539582 + 0.68341025j,
                0.61196844 + 0.70011239j,
                0.62997721 + 0.68103963j,
                0.63052723 + 0.67638061j,
                0.64753852 + 0.63785343j,
                0.68609065 + 0.61037295j,
                0.66114093 + 0.63164795j,
                0.64216788 + 0.64408527j,
                0.64083294 + 0.64792088j,
                0.6426059 + 0.65721866j,
                0.65127667 + 0.6693043j,
                0.64600835 + 0.68206702j,
                0.64201973 + 0.69240139j,
                0.63005618 + 0.70156604j,
                0.62493878 + 0.71293328j,
                0.63767207 + 0.71056997j,
                0.63744179 + 0.70486697j,
                0.64587089 + 0.69808462j,
                0.63909668 + 0.69082451j,
                0.64330159 + 0.68250018j,
                0.63464604 + 0.68888145j,
                0.62206823 + 0.70323509j,
                0.63991742 + 0.68542492j,
                0.6407161 + 0.68096328j,
                0.64231074 + 0.63093317j,
                0.68184858 + 0.61236309j,
                0.64975165 + 0.64633579j,
                0.62221087 + 0.67364908j,
                0.61869135 + 0.6786168j,
                0.61518713 + 0.69393699j,
                0.62558659 + 0.70295781j,
                0.62911769 + 0.70931058j,
                0.61475593 + 0.7311059j,
                0.59891595 + 0.74207498j,
                0.59714534 + 0.74298854j,
                0.6094539 + 0.73365856j,
                0.6142308 + 0.72040665j,
                0.62545701 + 0.71109305j,
                0.6249134 + 0.70090258j,
                0.64215679 + 0.68103574j,
                0.64289989 + 0.68170053j,
                0.63513016 + 0.68994668j,
                0.6589871 + 0.66945822j,
                0.67637484 + 0.65221818j,
                0.62878708 + 0.64595041j,
                0.66278383 + 0.6359361j,
                0.63927736 + 0.65127302j,
                0.62560969 + 0.67180777j,
                0.61813759 + 0.68316203j,
                0.60822895 + 0.70634477j,
                0.61913992 + 0.71381508j,
                0.62577007 + 0.71837007j,
                0.61418232 + 0.73695136j,
                0.60239077 + 0.74183236j,
                0.59788155 + 0.7511156j,
                0.60968238 + 0.73565489j,
                0.61486908 + 0.72053788j,
                0.63171507 + 0.70274674j,
                0.63956894 + 0.6874794j,
                0.65360317 + 0.6755892j,
                0.65151908 + 0.67754824j,
                0.64612916 + 0.68353672j,
                0.67192222 + 0.66238374j,
                0.67825473 + 0.66060121j,
                0.61999369 + 0.65642251j,
                0.64370469 + 0.65532714j,
                0.63832759 + 0.64897221j,
                0.62741077 + 0.66490695j,
                0.62525582 + 0.66807305j,
                0.60996569 + 0.6932607j,
                0.6228049 + 0.69594443j,
                0.62909265 + 0.70243554j,
                0.61719054 + 0.71955458j,
                0.61254668 + 0.71711797j,
                0.59385792 + 0.74276725j,
                0.60872812 + 0.721969j,
                0.60648405 + 0.71465645j,
                0.62440846 + 0.69660544j,
                0.63298797 + 0.68379446j,
                0.65141226 + 0.66697613j,
                0.65676056 + 0.66277912j,
                0.65133942 + 0.66772418j,
                0.68244267 + 0.64135943j,
                0.68447764 + 0.64756898j,
                0.59700378 + 0.69555812j,
                0.61326527 + 0.69011426j,
                0.63200713 + 0.66274255j,
                0.62392797 + 0.67631471j,
                0.63127352 + 0.67002698j,
                0.60577583 + 0.70045221j,
                0.62700912 + 0.69377619j,
                0.63026294 + 0.70164601j,
                0.61594615 + 0.7193101j,
                0.61285637 + 0.71612174j,
                0.58607768 + 0.75175806j,
                0.59155388 + 0.73313082j,
                0.57420146 + 0.74389625j,
                0.59187411 + 0.72844184j,
                0.59788267 + 0.71881963j,
                0.62651726 + 0.68924997j,
                0.63344168 + 0.68632373j,
                0.6311815 + 0.69235053j,
                0.65016939 + 0.67325376j,
                0.65854084 + 0.68231319j,
                0.61449226 + 0.69896238j,
                0.61317937 + 0.70487552j,
                0.64034165 + 0.66621509j,
                0.64312052 + 0.66502932j,
                0.64310655 + 0.66495986j,
                0.6213278 + 0.69055312j,
                0.63179382 + 0.68348563j,
                0.63994442 + 0.68540751j,
                0.63584861 + 0.69093988j,
                0.64130405 + 0.68052459j,
                0.60836499 + 0.72294731j,
                0.60406353 + 0.71723132j,
                0.59705487 + 0.71755922j,
                0.61513169 + 0.70233456j,
                0.61341404 + 0.7055794j,
                0.63540293 + 0.68113309j,
                0.64710003 + 0.6798608j,
                0.64718757 + 0.68393482j,
                0.66761985 + 0.66478289j,
                0.67789192 + 0.67271356j,
                0.5685101 + 0.71543434j,
                0.57241852 + 0.7219637j,
                0.61948901 + 0.669555j,
                0.61902173 + 0.67009714j,
                0.62408525 + 0.66552939j,
                0.6241032 + 0.67921299j,
                0.64204549 + 0.66698012j,
                0.65009144 + 0.6712098j,
                0.65021698 + 0.67446807j,
                0.66024396 + 0.65630759j,
                0.6100991 + 0.70886831j,
                0.60226204 + 0.71148318j,
                0.59577588 + 0.71424338j,
                0.60673079 + 0.70118505j,
                0.6061938 + 0.70073874j,
                0.62951847 + 0.67512093j,
                0.64234846 + 0.67329282j,
                0.63988727 + 0.68207514j,
                0.66876973 + 0.65946413j,
                0.69262507 + 0.65137772j,
                0.56920027 + 0.72357955j,
                0.56488941 + 0.73393103j,
                0.61018954 + 0.68122989j,
                0.61576066 + 0.67794544j,
                0.62083196 + 0.67234449j,
                0.62314609 + 0.681524j,
                0.64385574 + 0.664315j,
                0.66354557 + 0.65394672j,
                0.67007787 + 0.65543034j,
                0.68134121 + 0.64351793j,
                0.63217095 + 0.69810997j,
                0.61474084 + 0.70793884j,
                0.61901563 + 0.70259904j,
                0.62254827 + 0.69766439j,
                0.60986011 + 0.70782753j,
                0.63105836 + 0.68197441j,
                0.63968162 + 0.68231397j,
                0.64057199 + 0.68223903j,
                0.67311161 + 0.65349981j,
                0.70836446 + 0.635471j,
                0.53276034 + 0.76415634j,
                0.52561474 + 0.77618771j,
                0.57968115 + 0.71434735j,
                0.59380896 + 0.70454722j,
                0.60029569 + 0.69826706j,
                0.61082813 + 0.69516563j,
                0.63167873 + 0.67951783j,
                0.64954674 + 0.66977342j,
                0.66320704 + 0.66483853j,
                0.67249995 + 0.65200294j,
                0.6271045 + 0.7036974j,
                0.59532068 + 0.71905161j,
                0.58944194 + 0.7178904j,
                0.59160665 + 0.71313749j,
                0.58305747 + 0.71426438j,
                0.61136243 + 0.6818462j,
                0.62666938 + 0.66831424j,
                0.63362592 + 0.66171788j,
                0.66262694 + 0.62893128j,
                0.71737688 + 0.59668611j,
                0.5534531 + 0.75213496j,
                0.53279935 + 0.77528032j,
                0.57819369 + 0.72004203j,
                0.59228083 + 0.70818124j,
                0.59299442 + 0.70437577j,
                0.59264371 + 0.70715214j,
                0.6148803 + 0.69184373j,
                0.63056751 + 0.68216329j,
                0.64994233 + 0.6723401j,
                0.66726632 + 0.65151374j,
                0.64291642 + 0.68737258j,
                0.6178342 + 0.69778246j,
                0.61367624 + 0.69940439j,
                0.63086633 + 0.68298846j,
                0.62211872 + 0.68452284j,
                0.63463865 + 0.6659139j,
                0.64649174 + 0.64919637j,
                0.65415705 + 0.64269432j,
                0.66598408 + 0.62162097j,
                0.70922358 + 0.59881236j,
                0.59393224 + 0.73035691j,
                0.57194535 + 0.74566988j,
                0.60273106 + 0.71031378j,
                0.6119958 + 0.6974223j,
                0.60241279 + 0.69970776j,
                0.60824367 + 0.69694763j,
                0.61905717 + 0.68601386j,
                0.6194278 + 0.68675918j,
                0.62692694 + 0.68391269j,
                0.64664354 + 0.66100524j,
                0.62699627 + 0.68508096j,
                0.59356878 + 0.70928625j,
                0.59716315 + 0.70476541j,
                0.5974541 + 0.70185621j,
                0.59952408 + 0.69827279j,
                0.62153371 + 0.6690546j,
                0.64674855 + 0.63718985j,
                0.64808744 + 0.63873207j,
                0.65692186 + 0.62221642j,
                0.71066722 + 0.58260275j,
                0.61795883 + 0.69865207j,
                0.59453763 + 0.71782932j,
                0.62216846 + 0.68735902j,
                0.62457047 + 0.68186719j,
                0.6110587 + 0.69159259j,
                0.61263211 + 0.70108624j,
                0.62222267 + 0.6937708j,
                0.61373259 + 0.70157417j,
                0.61382672 + 0.70173763j,
                0.62108197 + 0.68750485j,
                0.61041473 + 0.70386164j,
                0.57736532 + 0.73212484j,
                0.57949327 + 0.72513993j,
                0.57803479 + 0.72243437j,
                0.57951881 + 0.71460174j,
                0.60825142 + 0.67887711j,
                0.63062424 + 0.65236315j,
                0.64061757 + 0.64887369j,
                0.6362831 + 0.64289355j,
                0.69440132 + 0.59142308j,
                0.63989919 + 0.66722005j,
                0.62088314 + 0.68945986j,
                0.63196635 + 0.67602258j,
                0.62833712 + 0.67409162j,
                0.61490193 + 0.6878096j,
                0.60171143 + 0.71130114j,
                0.59441063 + 0.71494141j,
                0.59182363 + 0.71791734j,
                0.60346564 + 0.71260244j,
                0.61148812 + 0.69730047j,
                0.61280386 + 0.70295179j,
                0.59791434 + 0.72252157j,
                0.60581365 + 0.70687299j,
                0.620001 + 0.69333516j,
                0.6366478 + 0.67666965j,
                0.6518266 + 0.65559753j,
                0.67126986 + 0.62500843j,
                0.6786309 + 0.62261954j,
                0.67824297 + 0.61339877j,
                0.71936725 + 0.56869245j,
                0.6206617 + 0.67555895j,
                0.61046639 + 0.69203555j,
                0.61685333 + 0.68278248j,
                0.6051204 + 0.69297215j,
                0.59626479 + 0.7036593j,
                0.58295118 + 0.7264538j,
                0.57258818 + 0.73723488j,
                0.57159263 + 0.74008082j,
                0.5803952 + 0.74059973j,
                0.57955081 + 0.73287034j,
                0.58656868 + 0.73238879j,
                0.58370242 + 0.73806412j,
                0.58376012 + 0.72878212j,
                0.59856311 + 0.71369738j,
                0.62226501 + 0.68538249j,
                0.63812124 + 0.66234027j,
                0.66328396 + 0.62210987j,
                0.67599961 + 0.6142984j,
                0.67470009 + 0.60556375j,
                0.70856211 + 0.56492085j,
                0.68696016 + 0.63393116j,
                0.65817565 + 0.66207473j,
                0.65063339 + 0.66106318j,
                0.64774688 + 0.66777628j,
                0.62629939 + 0.68933366j,
                0.59109014 + 0.72604526j,
                0.56572359 + 0.74568006j,
                0.55973692 + 0.75155626j,
                0.55796493 + 0.76024029j,
                0.55540519 + 0.75726715j,
                0.57981577 + 0.74639635j,
                0.56849056 + 0.75520616j,
                0.57940993 + 0.73589411j,
                0.59920246 + 0.7189598j,
                0.62052145 + 0.69965194j,
                0.64254622 + 0.66977469j,
                0.67270084 + 0.62200842j,
                0.6929185 + 0.60543036j,
                0.67756796 + 0.60476081j,
                0.70851811 + 0.56655157j,
                0.67796662 + 0.6397934j,
                0.65550862 + 0.66171991j,
                0.65450687 + 0.65389889j,
                0.64719062 + 0.66262334j,
                0.62904975 + 0.6836154j,
                0.58652258 + 0.73129151j,
                0.55178614 + 0.76069353j,
                0.53831149 + 0.77177785j,
                0.54085306 + 0.77248838j,
                0.52646661 + 0.77749377j,
                0.54815393 + 0.76855854j,
                0.54046631 + 0.77419499j,
                0.53853128 + 0.7637145j,
                0.56915145 + 0.73296819j,
                0.60546757 + 0.70256218j,
                0.63221499 + 0.66825021j,
                0.67383451 + 0.60737576j,
                0.70257932 + 0.58577018j,
                0.68737665 + 0.58586006j,
                0.71839694 + 0.5494397j,
            ]
        ).reshape((20, 20))
        self._coherence_bin_edges = np.array(
            [
                0.0,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.12,
                0.14,
                0.16,
                0.18,
                0.2,
                0.22,
                0.24,
                0.26,
                0.28,
                0.3,
                0.32,
                0.34,
                0.36,
                0.38,
                0.4,
                0.42,
                0.44,
                0.46,
                0.48,
                0.5,
                0.52,
                0.54,
                0.56,
                0.58,
                0.6,
                0.62,
                0.64,
                0.66,
                0.68,
                0.7,
                0.72,
                0.74,
                0.76,
                0.78,
                0.8,
                0.82,
                0.84,
                0.86,
                0.88,
                0.9,
                0.92,
                0.94,
                0.96,
                0.98,
                1.0,
            ]
        )
        self._az_hist = np.array(
            [
                [1, 0, 0, 3, 3, 0],
                [3, 1, 0, 1, 1, 1],
                [2, 1, 0, 1, 0, 4],
                [0, 2, 1, 4, 0, 1],
                [2, 0, 1, 1, 1, 1],
                [0, 2, 2, 0, 2, 1],
                [2, 5, 1, 0, 2, 1],
                [3, 0, 0, 0, 1, 2],
                [1, 0, 2, 3, 0, 3],
                [2, 0, 0, 2, 2, 0],
                [1, 2, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 2],
                [3, 1, 4, 3, 1, 0],
                [4, 2, 1, 1, 1, 2],
                [3, 3, 0, 0, 2, 4],
                [1, 4, 2, 0, 4, 0],
                [0, 4, 6, 3, 0, 2],
                [2, 0, 0, 1, 1, 0],
                [2, 2, 2, 2, 3, 0],
                [2, 1, 0, 0, 1, 1],
                [1, 2, 1, 3, 3, 1],
                [2, 2, 2, 0, 1, 0],
                [1, 1, 4, 1, 2, 0],
                [0, 1, 2, 3, 0, 3],
                [0, 2, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 2],
                [1, 1, 0, 0, 1, 0],
                [3, 1, 3, 1, 0, 2],
                [2, 2, 3, 2, 2, 3],
                [3, 0, 1, 0, 1, 0],
                [2, 0, 0, 0, 0, 1],
                [1, 1, 0, 4, 1, 0],
                [1, 2, 0, 2, 1, 1],
                [2, 3, 2, 0, 1, 0],
                [2, 4, 2, 0, 1, 0],
                [2, 3, 1, 2, 2, 2],
                [2, 5, 1, 3, 1, 0],
                [4, 1, 3, 0, 2, 4],
                [3, 1, 2, 1, 1, 1],
                [2, 2, 1, 1, 2, 2],
                [2, 0, 2, 0, 2, 2],
                [1, 1, 1, 0, 2, 1],
                [2, 3, 1, 4, 3, 0],
                [3, 1, 0, 1, 0, 0],
                [1, 2, 3, 0, 2, 2],
                [0, 1, 0, 1, 2, 3],
                [0, 3, 0, 0, 0, 0],
                [2, 1, 1, 1, 0, 2],
                [1, 1, 0, 0, 0, 1],
                [0, 2, 0, 1, 0, 1],
            ],
            dtype=np.int64,
        )
        self._rng_hist = np.array(
            [
                [2, 0, 1, 0, 0, 0, 3, 0, 1],
                [0, 3, 0, 0, 0, 1, 2, 1, 0],
                [2, 2, 1, 0, 0, 1, 0, 1, 1],
                [4, 1, 1, 0, 0, 2, 0, 0, 0],
                [2, 0, 0, 1, 0, 1, 0, 1, 1],
                [2, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 5, 1, 1, 1, 1, 0, 1],
                [1, 2, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 2, 0, 1, 2, 1],
                [1, 1, 1, 0, 0, 2, 0, 1, 0],
                [0, 0, 2, 0, 0, 0, 3, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [2, 4, 0, 0, 3, 0, 1, 0, 2],
                [3, 1, 0, 0, 1, 1, 3, 2, 0],
                [1, 1, 2, 1, 1, 5, 0, 0, 1],
                [2, 3, 0, 3, 1, 0, 0, 0, 2],
                [1, 1, 4, 2, 2, 2, 1, 1, 1],
                [1, 2, 0, 0, 1, 0, 0, 0, 0],
                [4, 0, 0, 2, 1, 0, 2, 2, 0],
                [1, 0, 0, 0, 2, 0, 1, 0, 1],
                [1, 1, 1, 3, 0, 2, 0, 1, 2],
                [1, 3, 0, 0, 0, 1, 0, 0, 2],
                [2, 0, 2, 2, 1, 1, 1, 0, 0],
                [0, 5, 1, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0, 0, 1, 0],
                [0, 2, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 2, 0, 1, 3, 1, 0, 1],
                [3, 0, 1, 2, 3, 1, 2, 2, 0],
                [0, 0, 2, 1, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 1, 1, 2],
                [1, 1, 0, 2, 0, 0, 1, 0, 2],
                [1, 1, 0, 1, 1, 0, 3, 1, 0],
                [0, 3, 0, 0, 1, 1, 2, 0, 2],
                [3, 2, 2, 1, 3, 0, 0, 0, 1],
                [2, 1, 0, 1, 2, 2, 1, 3, 0],
                [3, 2, 1, 2, 0, 0, 2, 3, 1],
                [1, 1, 2, 1, 0, 1, 0, 2, 1],
                [1, 0, 1, 2, 0, 0, 0, 3, 3],
                [1, 0, 0, 1, 1, 1, 0, 2, 2],
                [0, 1, 0, 2, 1, 2, 0, 0, 0],
                [1, 0, 2, 1, 2, 1, 0, 4, 2],
                [0, 2, 0, 0, 1, 0, 2, 0, 0],
                [1, 4, 0, 1, 1, 1, 1, 0, 1],
                [5, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 1, 2],
                [0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.int64,
        )

    def test_generate_coherence_graphs(self) -> None:
        """Testing generate_coherence_graphs"""
        data = InterferometricCoherenceOutput(
            channel_name="1",
            swath="S1",
            burst=0,
            polarization=SARPolarization.VH,
            coherence=self._coherence,
            coherence_histograms=InterferometricCoherence2DHistograms(
                coherence_bin_edges=self._coherence_bin_edges,
                azimuth_histogram=self._az_hist,
                range_histogram=self._rng_hist,
            ),
        )
        tag = "_".join([data.swath, data.polarization.name])
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            generate_coherence_graphs(data=data, output_dir=temp_dir)
            self.assertTrue(temp_dir.joinpath("coherence_magnitude_graph_" + tag + ".png").is_file())


if __name__ == "__main__":
    unittest.main()