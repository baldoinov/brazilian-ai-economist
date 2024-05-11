import numpy as np
from src.foundation.base.registry import Registry


class Landmark:
    """Base class for Landmark entity classes.

    Landmark classes describe the entities that exist exclusively in the environment
    world. In other words, they represent entities that should not be included in an
    agent's inventory and are only observable through observations from the
    spatial world.

    Landmark classes describe the following properties:
         ownable: If each instance of the landmark belongs to an agent. For example, a
            "House" is ownable and belongs to the agent that constructs it whereas
            "Water" is not ownable.
        solid: If the landmark creates a physical barrier to movement (that is,
            if agents are prevented from occupying cells with the landmark).
            Importantly, if the landmark is ownable, the agent that owns a given
            landmark can occupy its cell even if the landmark is solid.
    """

    name = None
    color = None  # array of RGB values [0 - 1]
    ownable = None
    solid = True  # Solid = Cannot be passed through
    # (unless it is owned by the agent trying to pass through)

    def __init__(self):
        assert self.name is not None
        assert self.color is not None
        assert self.ownable is not None

        # No agent can pass through this landmark
        self.blocking = self.solid and not self.ownable

        # Only the agent that owns this landmark can pass through it
        self.private = self.solid and self.ownable

        # This landmark does not belong to any agent and it does not inhibit movement
        self.public = not self.solid and not self.ownable


landmark_registry = Registry(Landmark)
