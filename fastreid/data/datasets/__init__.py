# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""


# Person re-id datasets
from .cuhk03 import CUHK03
from .DG_cuhk_sysu import DG_CUHK_SYSU
from .DG_cuhk02 import DG_CUHK02
from .DG_cuhk03_labeled import DG_CUHK03_labeled
from .DG_cuhk03_detected import DG_CUHK03_detected
from .dukemtmcreid import DukeMTMC
from .DG_dukemtmcreid import DG_DukeMTMC
from .market1501 import Market1501
from .DG_market1501 import DG_Market1501
from .msmt17 import MSMT17
from .AirportALERT import AirportALERT
from .iLIDS import iLIDS
from .pku import PKU
from .grid import GRID
from .prai import PRAI
from .prid import PRID
from .DG_prid import DG_PRID
from .DG_grid import DG_GRID
from .sensereid import SenseReID
from .sysu_mm import SYSU_mm
from .thermalworld import Thermalworld
from .pes3d import PeS3D
from .caviara import CAVIARa
from .viper import VIPeR
from .DG_viper import DG_VIPeR
from .DG_iLIDS import DG_iLIDS
from .lpw import LPW
from .shinpuhkan import Shinpuhkan
# Vehicle re-id datasets
from .veri import VeRi
from .veri_keypoint import VeRi_keypoint
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild
from .MyMarket1501 import MyMarket1501
from .MyDukeMTMC import MyDukeMTMC



__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
