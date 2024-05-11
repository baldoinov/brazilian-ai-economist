import lz4
import json
import numpy as np

from src.foundation.base.environment import BaseEnvironment


def annealed_tax_limit(completions, warmup_period, slope, final_max_tax_value=1.0):
    """
    Compute the maximum tax rate available at this stage of tax annealing.

    This function uses the number of episode completions and the annealing schedule
    (warmup_period, slope, & final_max_tax_value) to determine what the maximum tax
    rate can be.
    This type of annealing allows for a tax curriculum where earlier episodes are
    restricted to lower tax rates. As more episodes are played, higher tax values are
    allowed.

    Args:
        completions (int): Number of times the environment has completed an episode.
            Expected to be >= 0.
        warmup_period (int): Until warmup_period completions, only allow 0 tax. Using
            a negative value will enable non-0 taxes at 0 environment completions.
        slope (float): After warmup_period completions, percentage of full tax value
            unmasked with each new completion.
        final_max_tax_value (float): The maximum tax value at the end of annealing.

    Returns:
        A scalar value indicating the maximum tax at this stage of annealing.

    Example:
        >> WARMUP = 100
        >> SLOPE = 0.01
        >> annealed_tax_limit(0, WARMUP, SLOPE)
        0.0
        >> annealed_tax_limit(100, WARMUP, SLOPE)
        0.0
        >> annealed_tax_limit(150, WARMUP, SLOPE)
        0.5
        >> annealed_tax_limit(200, WARMUP, SLOPE)
        1.0
        >> annealed_tax_limit(1000, WARMUP, SLOPE)
        1.0
    """
    # What percentage of the full range is currently visible
    # (between 0 [only 0 tax] and 1 [all taxes visible])
    percentage_visible = np.maximum(
        0.0, np.minimum(1.0, slope * (completions - warmup_period))
    )

    # Determine the highest allowable tax,
    # given the current position in the annealing schedule
    current_max_tax = percentage_visible * final_max_tax_value

    return current_max_tax


def annealed_tax_mask(completions, warmup_period, slope, tax_values):
    """
    Generate a mask applied to a set of tax values for the purpose of tax annealing.

    This function uses the number of episode completions and the annealing schedule
    to determine which of the tax values are considered valid. The most extreme
    tax/subsidy values are unmasked last. Zero tax is always unmasked (i.e. always
    valid).
    This type of annealing allows for a tax curriculum where earlier episodes are
    restricted to lower tax rates. As more episodes are played, higher tax values are
    allowed.

    Args:
        completions (int): Number of times the environment has completed an episode.
            Expected to be >= 0.
        warmup_period (int): Until warmup_period completions, only allow 0 tax. Using
            a negative value will enable non-0 taxes at 0 environment completions.
        slope (float): After warmup_period completions, percentage of full tax value
            unmasked with each new completion.
        tax_values (list): The list of tax values associated with each action to
            which this mask will apply.

    Returns:
        A binary mask with same shape as tax_values, indicating which tax values are
            currently valid.

    Example:
        >> WARMUP = 100
        >> SLOPE = 0.01
        >> TAX_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
        >> annealed_tax_limit(0, WARMUP, SLOPE, TAX_VALUES)
        [0, 0, 0, 0, 0]
        >> annealed_tax_limit(100, WARMUP, SLOPE, TAX_VALUES)
        [0, 0, 0, 0, 0]
        >> annealed_tax_limit(150, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 0, 0]
        >> annealed_tax_limit(200, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 1, 1]
        >> annealed_tax_limit(1000, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 1, 1]
    """
    # Infer the most extreme tax level from the supplied tax values.
    abs_tax = np.abs(tax_values)
    full_tax_amount = np.max(abs_tax)

    # Determine the highest allowable tax, given the current position
    # in the annealing schedule
    max_absolute_visible_tax = annealed_tax_limit(
        completions, warmup_period, slope, full_tax_amount
    )

    # Return a binary mask to allow for taxes
    # at or below the highest absolute visible tax
    return np.less_equal(np.abs(tax_values), max_absolute_visible_tax).astype(
        np.float32
    )


def isoelastic_coin_minus_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def coin_minus_labor_cost(
    coin_endowment, total_labor, labor_exponent, labor_coefficient
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_exponent (float): Constant describing the shape of the utility profile
            with respect to total labor. Must be between >1.
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert labor_exponent > 1

    # Utility from coin endowment
    util_c = coin_endowment

    # Disutility from labor
    util_l = (total_labor**labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def coin_eq_times_productivity(coin_endowments, equality_weight):
    """Social welfare, measured as productivity scaled by the degree of coin equality.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        equality_weight (float): Constant that determines how productivity is scaled
            by coin equality. Must be between 0 (SW = prod) and 1 (SW = prod * eq).

    Returns:
        Product of coin equality and productivity (float).
    """
    n_agents = len(coin_endowments)
    prod = get_productivity(coin_endowments) / n_agents
    equality = equality_weight * get_equality(coin_endowments) + (1 - equality_weight)
    return equality * prod


def inv_income_weighted_coin_endowments(coin_endowments):
    """Social welfare, as weighted average endowment (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Weighted average coin endowment (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(coin_endowments * pareto_weights)


def inv_income_weighted_utility(coin_endowments, utilities):
    """Social welfare, as weighted average utility (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilities (ndarray): The array of utilities for each of the agents in the
            simulated economy.

    Returns:
        Weighted average utility (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(utilities * pareto_weights)


def get_gini(endowments):
    """Returns the normalized Gini index describing the distribution of endowments.

    https://en.wikipedia.org/wiki/Gini_coefficient

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized Gini index for the distribution of endowments (float). A value of 1
            indicates everything belongs to 1 agent (perfect inequality), whereas a
            value of 0 indicates all agents have equal endowments (perfect equality).

    Note:
        Uses a slightly different method depending on the number of agents. For fewer
        agents (<30), uses an exact but slow method. Switches to using a much faster
        method for more agents, where both methods produce approximately equivalent
        results.
    """
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


def get_equality(endowments):
    """Returns the complement of the normalized Gini index (equality = 1 - Gini).

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized equality index for the distribution of endowments (float). A value
            of 0 indicates everything belongs to 1 agent (perfect inequality),
            whereas a value of 1 indicates all agents have equal endowments (perfect
            equality).
    """
    return 1 - get_gini(endowments)


def get_productivity(coin_endowments):
    """Returns the total coin inside the simulated economy.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Total coin endowment (float).
    """
    return np.sum(coin_endowments)


def save_episode_log(game_object, filepath, compression_level=16):
    """Save a lz4 compressed version of the dense log stored
    in the provided game object"""
    assert isinstance(game_object, BaseEnvironment)
    compression_level = int(compression_level)
    if compression_level < 0:
        compression_level = 0
    elif compression_level > 16:
        compression_level = 16

    with lz4.frame.open(
        filepath, mode="wb", compression_level=compression_level
    ) as log_file:
        log_bytes = bytes(
            json.dumps(
                game_object.previous_episode_dense_log, ensure_ascii=False
            ).encode("utf-8")
        )
        log_file.write(log_bytes)


def load_episode_log(filepath):
    """Load the dense log saved at provided filepath"""
    with lz4.frame.open(filepath, mode="rb") as log_file:
        log_bytes = log_file.read()
    return json.loads(log_bytes)
