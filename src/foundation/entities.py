import numpy as np

from src.foundation.base.endogenous import Endogenous
from src.foundation.base.endogenous import endogenous_registry
from src.foundation.base.landmark import Landmark
from src.foundation.base.landmark import landmark_registry
from src.foundation.base.resource import Resource
from src.foundation.base.resource import resource_registry


@landmark_registry.add
class CoinSourceBlock(Landmark):
    """Special Landmark for generating resources. Not ownable. Not solid."""

    name = "CoinSourceBlock"
    color = np.array([229, 211, 82]) / 255.0
    ownable = False
    solid = False


@landmark_registry.add
class StoneSourceBlock(Landmark):
    """Special Landmark for generating resources. Not ownable. Not solid."""

    name = "StoneSourceBlock"
    color = np.array([241, 233, 219]) / 255.0
    ownable = False
    solid = False


@landmark_registry.add
class WoodSourceBlock(Landmark):
    """Special Landmark for generating resources. Not ownable. Not solid."""

    name = "WoodSourceBlock"
    color = np.array([107, 143, 113]) / 255.0
    ownable = False
    solid = False


@landmark_registry.add
class House(Landmark):
    """House landmark. Ownable. Solid."""

    name = "House"
    color = np.array([220, 20, 220]) / 255.0
    ownable = True
    solid = True


@landmark_registry.add
class Water(Landmark):
    """Water Landmark. Not ownable. Solid."""

    name = "Water"
    color = np.array([50, 50, 250]) / 255.0
    ownable = False
    solid = True


@resource_registry.add
class Wood(Resource):
    """Wood resource. collectible."""

    name = "Wood"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Stone(Resource):
    """Stone resource. collectible."""

    name = "Stone"
    color = np.array([241, 233, 219]) / 255.0
    collectible = True


@resource_registry.add
class Coin(Resource):
    """Coin resource. Included in all environments by default. Not collectible."""

    name = "Coin"
    color = np.array([229, 211, 82]) / 255.0
    collectible = False


@endogenous_registry.add
class Labor(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Labor"
