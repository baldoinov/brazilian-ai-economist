from src.foundation import utilities
from src.foundation.agents import agent_registry as agents
from src.foundation.components import component_registry as components
from src.foundation.entities import endogenous_registry as endogenous
from src.foundation.entities import landmark_registry as landmarks
from src.foundation.entities import resource_registry as resources
from src.foundation.scenarios import scenario_registry as scenarios


def make_env_instance(scenario_name, **kwargs):
    scenario_class = scenarios.get(scenario_name)
    return scenario_class(**kwargs)
