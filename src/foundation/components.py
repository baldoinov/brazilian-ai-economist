import numpy as np

from copy import deepcopy

from src.foundation.base.component import BaseComponent
from src.foundation.base.component import component_registry
from src.foundation.base.resource import resource_registry

from src.foundation.utilities import annealed_tax_limit
from src.foundation.utilities import annealed_tax_mask


@component_registry.add
class Build(BaseComponent):
    """
    Allows mobile agents to build house landmarks in the world using stone and wood,
    earning income.

    Can be configured to include heterogeneous building skill where agents earn
    different levels of income when building.

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int): Maximum skill multiplier that an agent
            can sample. Must be >= 1. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
        build_labor (float): Labor cost associated with building a house.
            Must be >= 0. Default is 10.
    """

    name = "Build"
    component_type = "Build"
    required_entities = ["Wood", "Stone", "Coin", "House", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        payment=10,
        payment_max_skill_multiplier=1,
        skill_dist="none",
        build_labor=10.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.payment_max_skill_multiplier = int(payment_max_skill_multiplier)
        assert self.payment_max_skill_multiplier >= 1

        self.resource_cost = {"Wood": 1, "Stone": 1}

        self.build_labor = float(build_labor)
        assert self.build_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills = {}

        self.builds = []

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicMobileAgent":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment": float(self.payment), "build_skill": 1}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        build = []
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action == 1:
                if self.agent_can_build(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = agent.loc
                    world.create_landmark("House", loc_r, loc_c, agent.idx)

                    # Receive payment for the house
                    agent.state["inventory"]["Coin"] += agent.state["build_payment"]

                    # Incur the labor cost for building
                    agent.state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["build_payment"]),
                        }
                    )

            else:
                raise ValueError

        self.builds.append(build)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "build_payment": agent.state["build_payment"] / self.payment,
                "build_skill": self.sampled_skills[agent.idx],
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_build(agent)])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for builds in self.builds:
            for build in builds:
                idx = build["builder"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_houses = np.sum(world.maps.get("House") > 0)
        out_dict["total_builds"] = num_houses

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """
        world = self.world

        self.sampled_skills = {agent.idx: 1 for agent in world.agents}

        PMSM = self.payment_max_skill_multiplier

        for agent in world.agents:
            if self.skill_dist == "none":
                sampled_skill = 1
                pay_rate = 1
            elif self.skill_dist == "pareto":
                sampled_skill = np.random.pareto(4)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill = np.random.lognormal(-1, 0.5)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            else:
                raise NotImplementedError

            agent.state["build_payment"] = float(pay_rate * self.payment)
            agent.state["build_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

        self.builds = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.

        """
        return self.builds


@component_registry.add
class ContinuousDoubleAuction(BaseComponent):
    """Allows mobile agents to buy/sell collectible resources with one another.

    Implements a commodity-exchange-style market where agents may sell a unit of
        resource by submitting an ask (saying the minimum it will accept in payment)
        or may buy a resource by submitting a bid (saying the maximum it will pay in
        exchange for a unit of a given resource).

    Args:
        max_bid_ask (int): Maximum amount of coin that an agent can bid or ask for.
            Must be >= 1. Default is 10 coin.
        order_labor (float): Amount of labor incurred when an agent creates an order.
            Must be >= 0. Default is 0.25.
        order_duration (int): Number of environment timesteps before an unfilled
            bid/ask expires. Must be >= 1. Default is 50 timesteps.
        max_num_orders (int, optional): Maximum number of bids + asks that an agent can
            have open for a given resource. Must be >= 1. Default is no limit to
            number of orders.
    """

    name = "ContinuousDoubleAuction"
    component_type = "Trade"
    required_entities = ["Coin", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *args,
        max_bid_ask=10,
        order_labor=0.25,
        order_duration=50,
        max_num_orders=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # The max amount (in coin) that an agent can bid/ask for 1 unit of a commodity
        self.max_bid_ask = int(max_bid_ask)
        assert self.max_bid_ask >= 1
        self.price_floor = 0
        self.price_ceiling = int(max_bid_ask)

        # The amount of time (in timesteps) that an order stays in the books
        # before it expires
        self.order_duration = int(order_duration)
        assert self.order_duration >= 1

        # The maximum number of bid+ask orders an agent can have open
        # for each type of commodity
        self.max_num_orders = int(max_num_orders or self.order_duration)
        assert self.max_num_orders >= 1

        # The labor cost associated with creating a bid or ask order

        self.order_labor = float(order_labor)
        self.order_labor = max(self.order_labor, 0.0)

        # Each collectible resource in the world can be traded via this component
        self.commodities = [
            r for r in self.world.resources if resource_registry.get(r).collectible
        ]

        # These get reset at the start of an episode:
        self.asks = {c: [] for c in self.commodities}
        self.bids = {c: [] for c in self.commodities}
        self.n_orders = {
            c: {i: 0 for i in range(self.n_agents)} for c in self.commodities
        }
        self.executed_trades = []
        self.price_history = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.bid_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.ask_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }

    # Convenience methods
    # -------------------

    def _price_zeros(self):
        if 1 + self.price_ceiling - self.price_floor <= 0:
            print("ERROR!", self.price_ceiling, self.price_floor)

        return np.zeros(1 + self.price_ceiling - self.price_floor)

    def available_asks(self, resource, agent):
        """
        Get a histogram of asks for resource to which agent could bid against.

        Args:
            resource (str): Name of the resource
            agent (BasicMobileAgent or None): Object of agent for which available
                asks are being queried. If None, all asks are considered available.

        Returns:
            ask_hist (ndarray): For each possible price level, the number of
                available asks.
        """
        if agent is None:
            a_idx = -1
        else:
            a_idx = agent.idx
        ask_hist = self._price_zeros()
        for i, h in self.ask_hists[resource].items():
            if a_idx != i:
                ask_hist += h
        return ask_hist

    def available_bids(self, resource, agent):
        """
        Get a histogram of bids for resource to which agent could ask against.

        Args:
            resource (str): Name of the resource
            agent (BasicMobileAgent or None): Object of agent for which available
                bids are being queried. If None, all bids are considered available.

        Returns:
            bid_hist (ndarray): For each possible price level, the number of
                available bids.
        """
        if agent is None:
            a_idx = -1
        else:
            a_idx = agent.idx
        bid_hist = self._price_zeros()
        for i, h in self.bid_hists[resource].items():
            if a_idx != i:
                bid_hist += h
        return bid_hist

    def can_bid(self, resource, agent):
        """If agent can submit a bid for resource."""
        return self.n_orders[resource][agent.idx] < self.max_num_orders

    def can_ask(self, resource, agent):
        """If agent can submit an ask for resource."""
        return (
            self.n_orders[resource][agent.idx] < self.max_num_orders
            and agent.state["inventory"][resource] > 0
        )

    # Core components for this market
    # -------------------------------

    def create_bid(self, resource, agent, max_payment):
        """Create a new bid for resource, with agent offering max_payment.

        On a successful trade, payment will be at most max_payment, possibly less.

        The agent places the bid coin into escrow so that it may not be spent on
        something else while the order exists.
        """

        # The agent is past the max number of orders
        # or doesn't have enough money, do nothing
        if (not self.can_bid(resource, agent)) or agent.state["inventory"][
            "Coin"
        ] < max_payment:
            return

        assert self.price_floor <= max_payment <= self.price_ceiling

        bid = {"buyer": agent.idx, "bid": int(max_payment), "bid_lifetime": 0}

        # Add this to the bid book
        self.bids[resource].append(bid)
        self.bid_hists[resource][bid["buyer"]][bid["bid"] - self.price_floor] += 1
        self.n_orders[resource][agent.idx] += 1

        # Set aside whatever money the agent is willing to pay
        # (will get excess back if price ends up being less)
        _ = agent.inventory_to_escrow("Coin", int(max_payment))

        # Incur the labor cost of creating an order
        agent.state["endogenous"]["Labor"] += self.order_labor

    def create_ask(self, resource, agent, min_income):
        """
        Create a new ask for resource, with agent asking for min_income.

        On a successful trade, income will be at least min_income, possibly more.

        The agent places one unit of resource into escrow so that it may not be used
        for something else while the order exists.
        """
        # The agent is past the max number of orders
        # or doesn't the resource it's trying to sell, do nothing
        if not self.can_ask(resource, agent):
            return

        # is there an upper limit?
        assert self.price_floor <= min_income <= self.price_ceiling

        ask = {"seller": agent.idx, "ask": int(min_income), "ask_lifetime": 0}

        # Add this to the ask book
        self.asks[resource].append(ask)
        self.ask_hists[resource][ask["seller"]][ask["ask"] - self.price_floor] += 1
        self.n_orders[resource][agent.idx] += 1

        # Set aside the resource the agent is willing to sell
        amount = agent.inventory_to_escrow(resource, 1)
        assert amount == 1

        # Incur the labor cost of creating an order
        agent.state["endogenous"]["Labor"] += self.order_labor

    def match_orders(self):
        """
        This implements the continuous double auction by identifying valid bid/ask
        pairs and executing trades accordingly.

        Higher (lower) bids (asks) are given priority over lower (higher) bids (asks).
        Trades are executed using the price of whichever bid/ask order was placed
        first: bid price if bid was placed first, ask price otherwise.

        Trading removes the payment and resource from bidder's and asker's escrow,
        respectively, and puts them in the other's inventory.
        """
        self.executed_trades.append([])

        for resource in self.commodities:
            possible_match = [True for _ in range(self.n_agents)]
            keep_checking = True

            bids = sorted(
                self.bids[resource],
                key=lambda b: (b["bid"], b["bid_lifetime"]),
                reverse=True,
            )
            asks = sorted(
                self.asks[resource], key=lambda a: (a["ask"], -a["ask_lifetime"])
            )

            while any(possible_match) and keep_checking:
                idx_bid, idx_ask = 0, 0
                while True:
                    # Out of bids to check. Exit both loops.
                    if idx_bid >= len(bids):
                        keep_checking = False
                        break

                    # Already know this buyer is no good for this round.
                    # Skip to next bid.
                    if not possible_match[bids[idx_bid]["buyer"]]:
                        idx_bid += 1

                    # Out of asks to check. This buyer won't find a match on this round.
                    # (maybe) Restart inner loop.
                    elif idx_ask >= len(asks):
                        possible_match[bids[idx_bid]["buyer"]] = False
                        break

                    # Skip to next ask if this ask comes from the buyer
                    # of the current bid.
                    elif asks[idx_ask]["seller"] == bids[idx_bid]["buyer"]:
                        idx_ask += 1

                    # If this bid/ask pair can't be matched, this buyer
                    # can't be matched. (maybe) Restart inner loop.
                    elif bids[idx_bid]["bid"] < asks[idx_ask]["ask"]:
                        possible_match[bids[idx_bid]["buyer"]] = False
                        break

                    # TRADE! (then restart inner loop)
                    else:
                        bid = bids.pop(idx_bid)
                        ask = asks.pop(idx_ask)

                        trade = {"commodity": resource}
                        trade.update(bid)
                        trade.update(ask)

                        if (
                            bid["bid_lifetime"] <= ask["ask_lifetime"]
                        ):  # Ask came earlier. (in other words,
                            # trade triggered by new bid)
                            trade["price"] = int(trade["ask"])
                        else:  # Bid came earlier. (in other words,
                            # trade triggered by new ask)
                            trade["price"] = int(trade["bid"])
                        trade["cost"] = trade["price"]  # What the buyer pays in total
                        trade["income"] = trade[
                            "price"
                        ]  # What the seller receives in total

                        buyer = self.world.agents[trade["buyer"]]
                        seller = self.world.agents[trade["seller"]]

                        # Bookkeeping
                        self.bid_hists[resource][bid["buyer"]][
                            bid["bid"] - self.price_floor
                        ] -= 1
                        self.ask_hists[resource][ask["seller"]][
                            ask["ask"] - self.price_floor
                        ] -= 1
                        self.n_orders[trade["commodity"]][seller.idx] -= 1
                        self.n_orders[trade["commodity"]][buyer.idx] -= 1
                        self.executed_trades[-1].append(trade)
                        self.price_history[resource][trade["seller"]][
                            trade["price"]
                        ] += 1

                        # The resource goes from the seller's escrow
                        # to the buyer's inventory
                        seller.state["escrow"][resource] -= 1
                        buyer.state["inventory"][resource] += 1

                        # Buyer's money (already set aside) leaves escrow
                        pre_payment = int(trade["bid"])
                        buyer.state["escrow"]["Coin"] -= pre_payment
                        assert buyer.state["escrow"]["Coin"] >= 0

                        # Payment is removed from the pre_payment
                        # and given to the seller. Excess returned to buyer.
                        payment_to_seller = int(trade["price"])
                        excess_payment_from_buyer = pre_payment - payment_to_seller
                        assert excess_payment_from_buyer >= 0
                        seller.state["inventory"]["Coin"] += payment_to_seller
                        buyer.state["inventory"]["Coin"] += excess_payment_from_buyer

                        # Restart the inner loop
                        break

            # Keep the unfilled bids/asks
            self.bids[resource] = bids
            self.asks[resource] = asks

    def remove_expired_orders(self):
        """
        Increment the time counter for any unfilled bids/asks and remove expired
        orders from the market.

        When orders expire, the payment or resource is removed from escrow and
        returned to the inventory and the associated order is removed from the order
        books.
        """
        world = self.world

        for resource in self.commodities:

            bids_ = []
            for bid in self.bids[resource]:
                bid["bid_lifetime"] += 1
                # If the bid is not expired, keep it in the bids
                if bid["bid_lifetime"] <= self.order_duration:
                    bids_.append(bid)
                # Otherwise, remove it and do the associated bookkeeping
                else:
                    # Return the set aside money to the buyer
                    amount = world.agents[bid["buyer"]].escrow_to_inventory(
                        "Coin", bid["bid"]
                    )
                    assert amount == bid["bid"]
                    # Adjust the bid histogram to reflect the removal of the bid
                    self.bid_hists[resource][bid["buyer"]][
                        bid["bid"] - self.price_floor
                    ] -= 1
                    # Adjust the order counter
                    self.n_orders[resource][bid["buyer"]] -= 1

            asks_ = []
            for ask in self.asks[resource]:
                ask["ask_lifetime"] += 1
                # If the ask is not expired, keep it in the asks
                if ask["ask_lifetime"] <= self.order_duration:
                    asks_.append(ask)
                # Otherwise, remove it and do the associated bookkeeping
                else:
                    # Return the set aside resource to the seller
                    resource_unit = world.agents[ask["seller"]].escrow_to_inventory(
                        resource, 1
                    )
                    assert resource_unit == 1
                    # Adjust the ask histogram to reflect the removal of the ask
                    self.ask_hists[resource][ask["seller"]][
                        ask["ask"] - self.price_floor
                    ] -= 1
                    # Adjust the order counter
                    self.n_orders[resource][ask["seller"]] -= 1

            self.bids[resource] = bids_
            self.asks[resource] = asks_

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Adds 2*C action spaces [ (bid+ask) * n_commodities ], each with 1 + max_bid_ask
        actions corresponding to price levels 0 to max_bid_ask.
        """
        # This component adds 2*(1+max_bid_ask)*n_resources possible actions:
        # buy/sell x each-price x each-resource
        if agent_cls_name == "BasicMobileAgent":
            trades = []
            for c in self.commodities:
                trades.append(
                    ("Buy_{}".format(c), 1 + self.max_bid_ask)
                )  # How much willing to pay for c
                trades.append(
                    ("Sell_{}".format(c), 1 + self.max_bid_ask)
                )  # How much need to receive to sell c
            return trades

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        """
        # This component doesn't add any state fields
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Create new bids and asks, match and execute valid order pairs, and manage
        order expiration.
        """
        world = self.world

        for resource in self.commodities:
            for agent in world.agents:
                self.price_history[resource][agent.idx] *= 0.995

                # Create bid action
                # -----------------
                resource_action = agent.get_component_action(
                    self.name, "Buy_{}".format(resource)
                )

                # No-op
                if resource_action == 0:
                    pass

                # Create a bid
                elif resource_action <= self.max_bid_ask + 1:
                    self.create_bid(resource, agent, max_payment=resource_action - 1)

                else:
                    raise ValueError

                # Create ask action
                # -----------------
                resource_action = agent.get_component_action(
                    self.name, "Sell_{}".format(resource)
                )

                # No-op
                if resource_action == 0:
                    pass

                # Create an ask
                elif resource_action <= self.max_bid_ask + 1:
                    self.create_ask(resource, agent, min_income=resource_action - 1)

                else:
                    raise ValueError

        # Here's where the magic happens:
        self.match_orders()  # Pair bids and asks
        self.remove_expired_orders()  # Get rid of orders that have expired

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents and the planner both observe historical market behavior and
        outstanding bids/asks for each tradable commodity. Agents only see the
        outstanding bids/asks to which they could respond (that is, that they did not
        submit). Agents also see their own outstanding bids/asks.
        """
        world = self.world

        obs = {a.idx: {} for a in world.agents + [world.planner]}

        prices = np.arange(self.price_floor, self.price_ceiling + 1)
        for c in self.commodities:
            net_price_history = np.sum(
                np.stack([self.price_history[c][i] for i in range(self.n_agents)]),
                axis=0,
            )
            market_rate = prices.dot(net_price_history) / np.maximum(
                0.001, np.sum(net_price_history)
            )
            scaled_price_history = net_price_history * self.inv_scale

            full_asks = self.available_asks(c, agent=None)
            full_bids = self.available_bids(c, agent=None)

            obs[world.planner.idx].update(
                {
                    "market_rate-{}".format(c): market_rate,
                    "price_history-{}".format(c): scaled_price_history,
                    "full_asks-{}".format(c): full_asks,
                    "full_bids-{}".format(c): full_bids,
                }
            )

            for _, agent in enumerate(world.agents):
                # Private to the agent
                obs[agent.idx].update(
                    {
                        "market_rate-{}".format(c): market_rate,
                        "price_history-{}".format(c): scaled_price_history,
                        "available_asks-{}".format(c): full_asks
                        - self.ask_hists[c][agent.idx],
                        "available_bids-{}".format(c): full_bids
                        - self.bid_hists[c][agent.idx],
                        "my_asks-{}".format(c): self.ask_hists[c][agent.idx],
                        "my_bids-{}".format(c): self.bid_hists[c][agent.idx],
                    }
                )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Agents cannot submit bids/asks for resources where they are at the order
        limit. In addition, they may only submit asks for resources they possess and
        bids for which they can pay.
        """
        world = self.world

        masks = dict()

        for agent in world.agents:
            masks[agent.idx] = {}

            can_pay = np.arange(self.max_bid_ask + 1) <= agent.inventory["Coin"]

            for resource in self.commodities:
                if not self.can_ask(resource, agent):  # asks_maxed:
                    masks[agent.idx]["Sell_{}".format(resource)] = np.zeros(
                        1 + self.max_bid_ask
                    )
                else:
                    masks[agent.idx]["Sell_{}".format(resource)] = np.ones(
                        1 + self.max_bid_ask
                    )

                if not self.can_bid(resource, agent):
                    masks[agent.idx]["Buy_{}".format(resource)] = np.zeros(
                        1 + self.max_bid_ask
                    )
                else:
                    masks[agent.idx]["Buy_{}".format(resource)] = can_pay.astype(
                        np.int32
                    )

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        trade_keys = ["price", "cost", "income"]

        selling_stats = {
            a.idx: {
                c: {k: 0 for k in trade_keys + ["n_sales"]} for c in self.commodities
            }
            for a in world.agents
        }
        buying_stats = {
            a.idx: {
                c: {k: 0 for k in trade_keys + ["n_sales"]} for c in self.commodities
            }
            for a in world.agents
        }

        n_trades = 0

        for trades in self.executed_trades:
            for trade in trades:
                n_trades += 1
                i_s, i_b, c = trade["seller"], trade["buyer"], trade["commodity"]
                selling_stats[i_s][c]["n_sales"] += 1
                buying_stats[i_b][c]["n_sales"] += 1
                for k in trade_keys:
                    selling_stats[i_s][c][k] += trade[k]
                    buying_stats[i_b][c][k] += trade[k]

        out_dict = {}
        for a in world.agents:
            for c in self.commodities:
                for stats, prefix in zip(
                    [selling_stats, buying_stats], ["Sell", "Buy"]
                ):
                    n = stats[a.idx][c]["n_sales"]
                    if n == 0:
                        for k in trade_keys:
                            stats[a.idx][c][k] = np.nan
                    else:
                        for k in trade_keys:
                            stats[a.idx][c][k] /= n

                    for k, v in stats[a.idx][c].items():
                        out_dict["{}/{}{}/{}".format(a.idx, prefix, c, k)] = v

        out_dict["n_trades"] = n_trades

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset the order books.
        """
        self.bids = {c: [] for c in self.commodities}
        self.asks = {c: [] for c in self.commodities}
        self.n_orders = {
            c: {i: 0 for i in range(self.n_agents)} for c in self.commodities
        }

        self.price_history = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.bid_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.ask_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }

        self.executed_trades = []

    def get_dense_log(self):
        """
        Log executed trades.

        Returns:
            trades (list): A list of trade events. Each entry corresponds to a single
                timestep and contains a description of any trades that occurred on
                that timestep.
        """
        return self.executed_trades


@component_registry.add
class Gather(BaseComponent):
    """
    Allows mobile agents to move around the world and collect resources and prevents
    agents from moving to invalid locations.

    Can be configured to include collection skill, where agents have heterogeneous
    probabilities of collecting bonus resources without additional labor cost.

    Args:
        move_labor (float): Labor cost associated with movement. Must be >= 0.
            Default is 1.0.
        collect_labor (float): Labor cost associated with collecting resources. This
            cost is added (in addition to any movement cost) when the agent lands on
            a tile that is populated with resources (triggering collection).
            Must be >= 0. Default is 1.0.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a bonus prob of 0. "pareto" and
            "lognormal" sample skills from the associated distributions.
    """

    name = "Gather"
    required_entities = ["Coin", "House", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        move_labor=1.0,
        collect_labor=1.0,
        skill_dist="none",
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.move_labor = float(move_labor)
        assert self.move_labor >= 0

        self.collect_labor = float(collect_labor)
        assert self.collect_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.gathers = []

        self._aidx = np.arange(self.n_agents)[:, None].repeat(4, axis=1)
        self._roff = np.array([[0, 0, -1, 1]])
        self._coff = np.array([[-1, 1, 0, 0]])

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Adds 4 actions (move up, down, left, or right) for mobile agents.
        """
        # This component adds 4 action that agents can take:
        # move up, down, left, or right
        if agent_cls_name == "BasicMobileAgent":
            return 4
        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state field for collection skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"bonus_gather_prob": 0.0}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Move to adjacent, unoccupied locations. Collect resources when moving to
        populated resource tiles, adding the resource to the agent's inventory and
        de-populating it from the tile.
        """
        world = self.world

        gathers = []
        for agent in world.get_random_order_agents():

            if self.name not in agent.action:
                return
            action = agent.get_component_action(self.name)

            r, c = [int(x) for x in agent.loc]

            if action == 0:  # NO-OP!
                new_r, new_c = r, c

            elif action <= 4:
                if action == 1:  # Left
                    new_r, new_c = r, c - 1
                elif action == 2:  # Right
                    new_r, new_c = r, c + 1
                elif action == 3:  # Up
                    new_r, new_c = r - 1, c
                else:  # action == 4, # Down
                    new_r, new_c = r + 1, c

                # Attempt to move the agent (if the new coordinates aren't accessible,
                # nothing will happen)
                new_r, new_c = world.set_agent_loc(agent, new_r, new_c)

                # If the agent did move, incur the labor cost of moving
                if (new_r != r) or (new_c != c):
                    agent.state["endogenous"]["Labor"] += self.move_labor

            else:
                raise ValueError

            for resource, health in world.location_resources(new_r, new_c).items():
                if health >= 1:
                    n_gathered = 1 + (np.random() < agent.state["bonus_gather_prob"])
                    agent.state["inventory"][resource] += n_gathered
                    world.consume_resource(resource, new_r, new_c)
                    # Incur the labor cost of collecting a resource
                    agent.state["endogenous"]["Labor"] += self.collect_labor
                    # Log the gather
                    gathers.append(
                        dict(
                            agent=agent.idx,
                            resource=resource,
                            n=n_gathered,
                            loc=[new_r, new_c],
                        )
                    )

        self.gathers.append(gathers)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their collection skill. The planner does not observe
        anything from this component.
        """
        return {
            str(agent.idx): {"bonus_gather_prob": agent.state["bonus_gather_prob"]}
            for agent in self.world.agents
        }

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent moving to adjacent tiles that are already occupied (or outside the
        boundaries of the world)
        """
        world = self.world

        coords = np.array([agent.loc for agent in world.agents])[:, :, None]
        ris = coords[:, 0] + self._roff + 1
        cis = coords[:, 1] + self._coff + 1

        occ = np.pad(world.maps.unoccupied, ((1, 1), (1, 1)))
        acc = np.pad(world.maps.accessibility, ((0, 0), (1, 1), (1, 1)))
        mask_array = np.logical_and(occ[ris, cis], acc[self._aidx, ris, cis]).astype(
            np.float32
        )

        masks = {agent.idx: mask_array[i] for i, agent in enumerate(world.agents)}

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' collection skills.
        """
        for agent in self.world.agents:
            if self.skill_dist == "none":
                bonus_rate = 0.0
            elif self.skill_dist == "pareto":
                bonus_rate = np.minimum(2, np.random.pareto(3)) / 2
            elif self.skill_dist == "lognormal":
                bonus_rate = np.minimum(2, np.random.lognormal(-2.022, 0.938)) / 2
            else:
                raise NotImplementedError
            agent.state["bonus_gather_prob"] = float(bonus_rate)

        self.gathers = []

    def get_dense_log(self):
        """
        Log resource collections.

        Returns:
            gathers (list): A list of gather events. Each entry corresponds to a single
                timestep and contains a description of any resource gathers that
                occurred on that timestep.

        """
        return self.gathers


@component_registry.add
class WealthRedistribution(BaseComponent):
    """Redistributes the total coin of the mobile agents as evenly as possible.

    Note:
        If this component is used, it should always be the last component in the order!
    """

    name = "WealthRedistribution"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        return

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Redistributes inventory coins so that all agents have equal coin endowment.
        """
        world = self.world

        # Divide coins evenly
        ic = np.array([agent.state["inventory"]["Coin"] for agent in world.agents])
        ec = np.array([agent.state["escrow"]["Coin"] for agent in world.agents])
        tc = np.sum(ic + ec)
        target_share = tc / self.n_agents
        for agent in world.agents:
            agent.state["inventory"]["Coin"] = float(target_share - ec[agent.idx])

        ic = np.array([agent.state["inventory"]["Coin"] for agent in world.agents])
        ec = np.array([agent.state["escrow"]["Coin"] for agent in world.agents])
        tc_next = np.sum(ic + ec)
        assert np.abs(tc - tc_next) < 1

    def generate_observations(self):
        """This component does not add any observations."""
        obs = {}
        return obs

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        return masks


@component_registry.add
class PeriodicBracketTax(BaseComponent):
    """Periodically collect income taxes from agents and do lump-sum redistribution.

    Note:
        If this component is used, it should always be the last component in the order!

    Args:
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "saez" uses an adaptation of the theoretical optimal taxation formula
            derived in https://www.nber.org/papers/w7628.
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        pareto_weight_type (str): Type of Pareto weights to use when computing tax
            rates using the Saez formula. "inverse_income" (default) uses 1/z;
            "uniform" uses 1.
        saez_fixed_elas (float, optional): If supplied, this value will be used as
            the elasticity estimate when computing tax rates using the Saez formula.
            If not given (default), elasticity will be estimated empirically.
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.
    """

    name = "PeriodicBracketTax"
    component_type = "PeriodicTax"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        disable_taxes=False,
        tax_model="model_wrapper",
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=5,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing="us-federal",
        fixed_bracket_rates=None,
        pareto_weight_type="inverse_income",
        saez_fixed_elas=None,
        tax_annealing_schedule=None,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Whether to turn off taxes. Disabling taxes will prevent any taxes from
        # being collected but the observation space will be the same as if taxes were
        # enabled, which can be useful for controlled tax/no-tax comparisons.
        self.disable_taxes = bool(disable_taxes)

        # How to set taxes.
        self.tax_model = tax_model
        assert self.tax_model in [
            "model_wrapper",
            "us-federal-single-filer-2018-scaled",
            "saez",
            "fixed-bracket-rates",
        ]

        # How many timesteps a tax period lasts.
        self.period = int(period)
        assert self.period > 0

        # Minimum marginal bracket rate
        self.rate_min = 0.0 if self.disable_taxes else float(rate_min)
        # Maximum marginal bracket rate
        self.rate_max = 0.0 if self.disable_taxes else float(rate_max)
        assert 0 <= self.rate_min <= self.rate_max <= 1.0

        # Interval for discretizing tax rate options
        # (only applies if tax_model == "model_wrapper").
        self.rate_disc = float(rate_disc)

        self.use_discretized_rates = self.tax_model == "model_wrapper"

        if self.use_discretized_rates:
            self.disc_rates = np.arange(
                self.rate_min, self.rate_max + self.rate_disc, self.rate_disc
            )
            self.disc_rates = self.disc_rates[self.disc_rates <= self.rate_max]
            assert len(self.disc_rates) > 1 or self.disable_taxes
            self.n_disc_rates = len(self.disc_rates)
        else:
            self.disc_rates = None
            self.n_disc_rates = 0

        # === income bracket definitions ===
        self.n_brackets = int(n_brackets)
        assert self.n_brackets >= 2

        self.top_bracket_cutoff = float(top_bracket_cutoff)
        assert self.top_bracket_cutoff >= 10

        self.usd_scale = float(usd_scaling)
        assert self.usd_scale > 0

        self.bracket_spacing = bracket_spacing.lower()
        assert self.bracket_spacing in ["linear", "log", "us-federal"]

        if self.bracket_spacing == "linear":
            self.bracket_cutoffs = np.linspace(
                0, self.top_bracket_cutoff, self.n_brackets
            )

        elif self.bracket_spacing == "log":
            b0_max = self.top_bracket_cutoff / (2 ** (self.n_brackets - 2))
            self.bracket_cutoffs = np.concatenate(
                [
                    [0],
                    2
                    ** np.linspace(
                        np.log2(b0_max),
                        np.log2(self.top_bracket_cutoff),
                        n_brackets - 1,
                    ),
                ]
            )
        elif self.bracket_spacing == "us-federal":
            self.bracket_cutoffs = (
                np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])
                / self.usd_scale
            )
            self.n_brackets = len(self.bracket_cutoffs)
            self.top_bracket_cutoff = float(self.bracket_cutoffs[-1])
        else:
            raise NotImplementedError

        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]

        assert self.bracket_cutoffs[0] == 0

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            assert self.bracket_spacing == "us-federal"

        if self.tax_model == "fixed-bracket-rates":
            assert isinstance(fixed_bracket_rates, (tuple, list))
            assert np.min(fixed_bracket_rates) >= 0
            assert np.max(fixed_bracket_rates) <= 1
            assert len(fixed_bracket_rates) == self.n_brackets
            self._fixed_bracket_rates = np.array(fixed_bracket_rates)
        else:
            self._fixed_bracket_rates = None

        # === bracket tax rates ===
        self.curr_bracket_tax_rates = np.zeros_like(self.bracket_cutoffs)
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        # === Pareto weights, elasticity ===
        self.pareto_weight_type = pareto_weight_type
        self.elas_tm1 = 0.5
        self.elas_t = 0.5
        self.log_z0_tm1 = 0
        self.log_z0_t = 0

        self._saez_fixed_elas = saez_fixed_elas
        if self._saez_fixed_elas is not None:
            self._saez_fixed_elas = float(self._saez_fixed_elas)
            assert self._saez_fixed_elas >= 0

        # Size of the local buffer. In a distributed context, the global buffer size
        # will be capped at n_replicas * _buffer_size.
        # NOTE: Saez will use random taxes until it has self._buffer_size samples.
        self._buffer_size = 500
        self._reached_min_samples = False
        self._additions_this_episode = 0
        # Local buffer maintained by this replica.
        self._local_saez_buffer = []
        # "Global" buffer obtained by combining local buffers of individual replicas.
        self._global_saez_buffer = []

        self._saez_n_estimation_bins = 100
        self._saez_top_rate_cutoff = self.bracket_cutoffs[-1]
        self._saez_income_bin_edges = np.linspace(
            0, self._saez_top_rate_cutoff, self._saez_n_estimation_bins + 1
        )
        self._saez_income_bin_sizes = np.concatenate(
            [
                self._saez_income_bin_edges[1:] - self._saez_income_bin_edges[:-1],
                [np.inf],
            ]
        )
        self.running_avg_tax_rates = np.zeros_like(self.curr_bracket_tax_rates)

        # === tax cycle definitions ===
        self.tax_cycle_pos = 1
        self.last_coin = [0 for _ in range(self.n_agents)]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        # === trackers ===
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [0] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self.taxes = []

        # === tax annealing ===
        # for annealing of non-planner max taxes.
        self._annealed_rate_max = float(self.rate_max)
        self._last_completions = 0

        # for annealing of planner actions.
        self.tax_annealing_schedule = tax_annealing_schedule
        if tax_annealing_schedule is not None:
            assert isinstance(self.tax_annealing_schedule, (tuple, list))
            self._annealing_warmup = self.tax_annealing_schedule[0]
            self._annealing_slope = self.tax_annealing_schedule[1]
            self._annealed_rate_max = annealed_tax_limit(
                self._last_completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )
        else:
            self._annealing_warmup = None
            self._annealing_slope = None

        if self.tax_model == "model_wrapper" and not self.disable_taxes:
            planner_action_tuples = self.get_n_actions("BasicPlanner")
            self._planner_tax_val_dict = {
                k: self.disc_rates for k, v in planner_action_tuples
            }
        else:
            self._planner_tax_val_dict = {}
        self._planner_masks = None

        # === placeholders ===
        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

    # Methods for getting/setting marginal tax rates
    # ----------------------------------------------

    # ------- fixed-bracket-rates
    @property
    def fixed_bracket_rates(self):
        """Return whatever fixed bracket rates were set during initialization."""
        return self._fixed_bracket_rates

    @property
    def curr_rate_max(self):
        """Maximum allowable tax rate, given current progress of any tax annealing."""
        if self.tax_annealing_schedule is None:
            return self.rate_max
        return self._annealed_rate_max

    @property
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        if self.use_discretized_rates:
            return self.disc_rates[self.curr_rate_indices]

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            marginal_tax_bracket_rates = np.minimum(
                np.array(self.us_federal_single_filer_2018_scaled), self.curr_rate_max
            )
        elif self.tax_model == "saez":
            marginal_tax_bracket_rates = np.minimum(
                self.curr_bracket_tax_rates, self.curr_rate_max
            )
        elif self.tax_model == "fixed-bracket-rates":
            marginal_tax_bracket_rates = np.minimum(
                np.array(self.fixed_bracket_rates), self.curr_rate_max
            )
        else:
            raise NotImplementedError

        return marginal_tax_bracket_rates

    def set_new_period_rates_model(self):
        """Update taxes using actions from the tax model."""
        if self.disable_taxes:
            return

        # AI version
        for i, bracket in enumerate(self.bracket_cutoffs):
            planner_action = self.world.planner.get_component_action(
                self.name, "TaxIndexBracket_{:03d}".format(int(bracket))
            )
            if planner_action == 0:
                pass
            elif planner_action <= self.n_disc_rates:
                self.curr_rate_indices[i] = int(planner_action - 1)
            else:
                raise ValueError

    # ------- Saez formula
    def compute_and_set_new_period_rates_from_saez_formula(
        self, update_elas_tm1=True, update_log_z0_tm1=True
    ):
        """Estimates/sets optimal rates using adaptation of Saez formula

        See: https://www.nber.org/papers/w7628
        """
        # Until we reach the min sample number, keep checking if we have reached it.
        if not self._reached_min_samples:
            # Note: self.saez_buffer includes the global buffer (if applicable).
            if len(self.saez_buffer) >= self._buffer_size:
                self._reached_min_samples = True

        # If no enough samples, use random taxes.
        if not self._reached_min_samples:
            self.curr_bracket_tax_rates = np.random.uniform(
                low=self.rate_min,
                high=self.curr_rate_max,
                size=self.curr_bracket_tax_rates.shape,
            )
            return

        incomes_and_marginal_rates = np.array(self.saez_buffer)

        # Elasticity assumed constant for all incomes.
        # (Run this for the sake of tracking the estimate; will not actually use the
        # estimate if using fixed elasticity).
        if update_elas_tm1:
            self.elas_tm1 = float(self.elas_t)
        if update_log_z0_tm1:
            self.log_z0_tm1 = float(self.log_z0_t)

        elas_t, log_z0_t = self.estimate_uniform_income_elasticity(
            incomes_and_marginal_rates,
            elas_df=0.98,
            elas_tm1=self.elas_tm1,
            log_z0_tm1=self.log_z0_tm1,
            verbose=False,
        )

        if update_elas_tm1:
            self.elas_t = float(elas_t)
        if update_log_z0_tm1:
            self.log_z0_t = float(log_z0_t)

        # If a fixed estimate has been specified, use it in the formulas below.
        if self._saez_fixed_elas is not None:
            elas_t = float(self._saez_fixed_elas)

        # Get Saez parameters at each income bin
        # to compute a marginal tax rate schedule.
        binned_gzs, binned_azs = self.get_binned_saez_welfare_weight_and_pareto_params(
            population_incomes=incomes_and_marginal_rates[:, 0]
        )

        # Use the elasticity to compute this binned schedule using the Saez formula.
        binned_marginal_tax_rates = self.get_saez_marginal_rates(
            binned_gzs, binned_azs, elas_t
        )

        # Adapt the saez tax schedule to the tax brackets.
        self.curr_bracket_tax_rates = np.clip(
            self.bracketize_schedule(
                bin_marginal_rates=binned_marginal_tax_rates,
                bin_edges=self._saez_income_bin_edges,
                bin_sizes=self._saez_income_bin_sizes,
            ),
            self.rate_min,
            self.curr_rate_max,
        )

        self.running_avg_tax_rates = (self.running_avg_tax_rates * 0.99) + (
            self.curr_bracket_tax_rates * 0.01
        )

    # Implementation of the Saez formula in this periodic, bracketed setting
    # ----------------------------------------------------------------------
    @property
    def saez_buffer(self):
        if not self._global_saez_buffer:
            saez_buffer = self._local_saez_buffer
        elif self._additions_this_episode == 0:
            saez_buffer = self._global_saez_buffer
        else:
            saez_buffer = (
                self._global_saez_buffer
                + self._local_saez_buffer[-self._additions_this_episode :]
            )
        return saez_buffer

    def get_local_saez_buffer(self):
        return self._local_saez_buffer

    def set_global_saez_buffer(self, global_saez_buffer):
        assert isinstance(global_saez_buffer, list)
        assert len(global_saez_buffer) >= len(self._local_saez_buffer)
        self._global_saez_buffer = global_saez_buffer

    def _update_saez_buffer(self, tax_info_t):
        # Update the buffer.
        for a_idx in range(self.n_agents):
            z_t = tax_info_t[str(a_idx)]["income"]
            tau_t = tax_info_t[str(a_idx)]["marginal_rate"]
            self._local_saez_buffer.append([z_t, tau_t])
            self._additions_this_episode += 1

        while len(self._local_saez_buffer) > self._buffer_size:
            _ = self._local_saez_buffer.pop(0)

    def reset_saez_buffers(self):
        self._local_saez_buffer = []
        self._global_saez_buffer = []
        self._additions_this_episode = 0
        self._reached_min_samples = False

    def estimate_uniform_income_elasticity(
        self,
        observed_incomes_and_marginal_rates,
        elas_df=0.98,
        elas_tm1=0.5,
        log_z0_tm1=0.5,
        verbose=False,
    ):
        """Estimate elasticity using Ordinary Least Squares regression.
        OLS: https://en.wikipedia.org/wiki/Ordinary_least_squares
        Estimating elasticity: https://www.nber.org/papers/w7512
        """
        zs = []
        taus = []

        for z_t, tau_t in observed_incomes_and_marginal_rates:
            # If z_t is <=0 or tau_t is >=1, the operations below will give us nans
            if z_t > 0 and tau_t < 1:
                zs.append(z_t)
                taus.append(tau_t)

        if len(zs) < 10:
            return float(elas_tm1), float(log_z0_tm1)
        if np.std(taus) < 1e-6:
            return float(elas_tm1), float(log_z0_tm1)

        # Regressing log income against log 1-marginal_rate.
        x = np.log(np.maximum(1 - np.array(taus), 1e-9))
        # (bias term)
        b = np.ones_like(x)
        # Perform OLS.
        X = np.stack([x, b]).T  # Stack linear & bias terms
        Y = np.log(np.maximum(np.array(zs), 1e-9))  # Regression targets
        XXi = np.linalg.inv(X.T.dot(X))
        XY = X.T.dot(Y)
        elas, log_z0 = XXi.T.dot(XY)

        warn_less_than_0 = elas < 0
        instant_elas_t = np.maximum(elas, 0.0)

        elas_t = ((1 - elas_df) * instant_elas_t) + (elas_df * elas_tm1)

        if verbose:
            if warn_less_than_0:
                print("\nWARNING: Recent elasticity estimate is < 0.")
                print("Running elasticity estimate: {:.2f}\n".format(elas_t))
            else:
                print("\nRunning elasticity estimate: {:.2f}\n".format(elas_t))

        return elas_t, log_z0

    def get_binned_saez_welfare_weight_and_pareto_params(self, population_incomes):
        def clip(x, lo=None, hi=None):
            if lo is not None:
                x = max(lo, x)
            if hi is not None:
                x = min(x, hi)
            return x

        def bin_z(left, right):
            return 0.5 * (left + right)

        def get_cumul(counts, incomes_below, incomes_above):
            n_below = len(incomes_below)
            n_above = len(incomes_above)
            n_total = np.sum(counts) + n_below + n_above

            def p(i, counts):
                return counts[i] / n_total

            # Probability that an income is below the taxable threshold.
            p_below = n_below / n_total

            # pz = p(z' = z): probability that [binned] income z' occurs in bin z.
            pz = [p(i, counts) for i in range(len(counts))] + [n_above / n_total]

            # Pz = p(z' <= z): Probability z' is less-than or equal to z.
            cum_pz = [pz[0] + p_below]
            for p in pz[1:]:
                cum_pz.append(clip(cum_pz[-1] + p, 0, 1.0))

            return np.array(pz), np.array(cum_pz)

        def compute_binned_g_distribution(counts, lefts, incomes):
            def pareto(z):
                if self.pareto_weight_type == "uniform":
                    pareto_weights = np.ones_like(z)
                elif self.pareto_weight_type == "inverse_income":
                    pareto_weights = 1.0 / np.maximum(1, z)
                else:
                    raise NotImplementedError
                return pareto_weights

            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # The total (unnormalized) Pareto weight of untaxable incomes.
            if len(incomes_below) > 0:
                pareto_weight_below = np.sum(pareto(np.maximum(incomes_below, 0)))
            else:
                pareto_weight_below = 0

            # The total (unnormalized) Pareto weight within each bin.
            if len(incomes_above) > 0:
                pareto_weight_above = np.sum(pareto(incomes_above))
            else:
                pareto_weight_above = 0

            # The total (unnormalized) Pareto weight within each bin.
            pareto_weight_per_bin = counts * pareto(bin_z(lefts[:-1], lefts[1:]))

            # The aggregate (unnormalized) Pareto weight of all incomes.
            cumulative_pareto_weights = pareto_weight_per_bin.sum()
            cumulative_pareto_weights += pareto_weight_below
            cumulative_pareto_weights += pareto_weight_above

            # Normalize so that the Pareto density sums to 1.
            pareto_norm = cumulative_pareto_weights + 1e-9
            unnormalized_pareto_density = np.concatenate(
                [pareto_weight_per_bin, [pareto_weight_above]]
            )
            normalized_pareto_density = unnormalized_pareto_density / pareto_norm

            # Aggregate Pareto weight of earners with income greater-than or equal to z.
            cumulative_pareto_density_geq_z = np.cumsum(
                normalized_pareto_density[::-1]
            )[::-1]

            # Probability that [binned] income z' is greather-than or equal to z.
            pz, _ = get_cumul(counts, incomes_below, incomes_above)
            cumulative_prob_geq_z = np.cumsum(pz[::-1])[::-1]

            # Average (normalized) Pareto weight of earners with income >= z.
            geq_z_norm = cumulative_prob_geq_z + 1e-9
            avg_pareto_weight_geq_z = cumulative_pareto_density_geq_z / geq_z_norm

            def interpolate_gzs(gz):
                # Assume incomes within a bin are evenly distributed within that bin
                # and re-compute accordingly.
                gz_at_left_edge = gz[:-1]
                gz_at_right_edge = gz[1:]

                avg_bin_gz = 0.5 * (gz_at_left_edge + gz_at_right_edge)
                # Re-attach the gz of the top tax rate (does not need to be
                # interpolated).
                gzs = np.concatenate([avg_bin_gz, [gz[-1]]])
                return gzs

            return interpolate_gzs(avg_pareto_weight_geq_z)

        def compute_binned_a_distribution(counts, lefts, incomes):
            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # z is defined as the MIDDLE point in a bin.
            # So for a bin [left, right] -> z = (left + right) / 2.
            Az = []

            # cum_pz = p(z' <= z): Probability z' is less-than or equal to z
            pz, cum_pz = get_cumul(counts, incomes_below, incomes_above)

            # Probability z' is greater-than or equal to z
            # Note: The "0.5" coefficient gives results more consistent with theory; it
            # accounts for the assumption that incomes within a particular bin are
            # uniformly spread between the left & right edges of that bin.
            p_geq_z = 1 - cum_pz + (0.5 * pz)

            T = len(lefts[:-1])

            for i in range(T):
                if pz[i] == 0:
                    Az.append(np.nan)
                else:
                    z = bin_z(lefts[i], lefts[i + 1])
                    # paz = z * pz[i] / (clip(1 - Pz[i], 0, 1) + 1e-9)
                    paz = z * pz[i] / (clip(p_geq_z[i], 0, 1) + 1e-9)  # defn of A(z)
                    paz = paz / (lefts[i + 1] - lefts[i])  # norm by bin width
                    Az.append(paz)

            # Az for the incomes past the top cutoff,
            # the bin is [left, infinity]: there is no "middle".
            # Hence, use the mean value in the last bin.
            if len(incomes_above) > 0:
                cutoff = lefts[-1]
                avg_income_above_cutoff = np.mean(incomes_above)
                # use a special formula to compute A(z)
                Az_above = avg_income_above_cutoff / (
                    avg_income_above_cutoff - cutoff + 1e-9
                )
            else:
                Az_above = 0.0

            return np.concatenate([Az, [Az_above]])

        counts, lefts = np.histogram(
            population_incomes, bins=self._saez_income_bin_edges
        )
        population_gz = compute_binned_g_distribution(counts, lefts, population_incomes)
        population_az = compute_binned_a_distribution(counts, lefts, population_incomes)

        # Return the binned stats used to create a schedule of marginal rates.
        return population_gz, population_az

    @staticmethod
    def get_saez_marginal_rates(binned_gz, binned_az, elas, interpolate=True):
        # Marginal rates within each income bin (last tau is the top tax rate).
        taus = (1.0 - binned_gz) / (1.0 - binned_gz + binned_az * elas + 1e-9)

        if interpolate:
            # In bins where there were no incomes found, tau is nan.
            # Interpolate to fill the gaps.
            last_real_rate = 0.0
            last_real_tidx = -1
            for i, tau in enumerate(taus):
                # The current tax rate is a real number.
                if not np.isnan(tau):
                    # This is the end of a gap. Interpolate.
                    if (i - last_real_tidx) > 1:
                        assert (
                            i != 0
                        )  # This should never trigger for the first tax bin.
                        gap_indices = list(range(last_real_tidx + 1, i))
                        intermediate_rates = np.linspace(
                            last_real_rate, tau, len(gap_indices) + 2
                        )[1:-1]
                        assert len(gap_indices) == len(intermediate_rates)
                        for gap_index, intermediate_rate in zip(
                            gap_indices, intermediate_rates
                        ):
                            taus[gap_index] = intermediate_rate
                    # Update the tracker.
                    last_real_rate = float(tau)
                    last_real_tidx = int(i)

                # The current tax rate is a nan. Continue without updating
                # the tracker (indicating the presence of a gap).
                else:
                    pass

        return taus

    def bracketize_schedule(self, bin_marginal_rates, bin_edges, bin_sizes):
        # Compute the amount of tax each bracket would collect
        # if income was >= the right edge.
        # Divide by the bracket size to get
        # the average marginal rate within that bracket.
        last_bracket_total = 0
        bracket_avg_marginal_rates = []
        for b_idx, income in enumerate(self.bracket_cutoffs[1:]):
            # How much income occurs within each bin
            # (including the open-ended, top "bin").
            past_cutoff = np.maximum(0, income - bin_edges)
            bin_income = np.minimum(bin_sizes, past_cutoff)

            # To get the total taxes due,
            # multiply the income within each bin by that bin's marginal rate.
            bin_taxes = bin_marginal_rates * bin_income
            taxes_due = np.maximum(0, np.sum(bin_taxes))

            bracket_tax_burden = taxes_due - last_bracket_total
            bracket_size = self.bracket_sizes[b_idx]

            bracket_avg_marginal_rates.append(bracket_tax_burden / bracket_size)
            last_bracket_total = taxes_due

        # The top bracket tax rate is computed directly already.
        bracket_avg_marginal_rates.append(bin_marginal_rates[-1])

        bracket_rates = np.array(bracket_avg_marginal_rates)
        assert len(bracket_rates) == self.n_brackets

        return bracket_rates

    # Methods for collecting and redistributing taxes
    # -----------------------------------------------

    def income_bin(self, income):
        """Return index of tax bin in which income falls."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.bracket_cutoffs[np.argmax(bracket_bool)]

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    def taxes_due(self, income):
        """Return the total amount of taxes due at this income level."""
        past_cutoff = np.maximum(0, income - self.bracket_cutoffs)
        bin_income = np.minimum(self.bracket_sizes, past_cutoff)
        bin_taxes = self.curr_marginal_rates * bin_income
        return np.sum(bin_taxes)

    def enact_taxes(self):
        """Calculate period income & tax burden. Collect taxes and redistribute."""
        net_tax_revenue = 0
        tax_dict = dict(
            schedule=np.array(self.curr_marginal_rates),
            cutoffs=np.array(self.bracket_cutoffs),
        )

        for curr_rate, bracket_cutoff in zip(
            self.curr_marginal_rates, self.bracket_cutoffs
        ):
            self._schedules["{:03d}".format(int(bracket_cutoff))].append(
                float(curr_rate)
            )

        self.last_income = []
        self.last_effective_tax_rate = []
        self.last_marginal_rate = []
        for agent, last_coin in zip(self.world.agents, self.last_coin):
            income = agent.total_endowment("Coin") - last_coin
            tax_due = self.taxes_due(income)
            effective_taxes = np.minimum(
                agent.state["inventory"]["Coin"], tax_due
            )  # Don't take from escrow.
            marginal_rate = self.marginal_rate(income)
            effective_tax_rate = float(effective_taxes / np.maximum(0.000001, income))
            tax_dict[str(agent.idx)] = dict(
                income=float(income),
                tax_paid=float(effective_taxes),
                marginal_rate=marginal_rate,
                effective_rate=effective_tax_rate,
            )

            # Actually collect the taxes.
            agent.state["inventory"]["Coin"] -= effective_taxes
            net_tax_revenue += effective_taxes

            self.last_income.append(float(income))
            self.last_marginal_rate.append(float(marginal_rate))
            self.last_effective_tax_rate.append(effective_tax_rate)

            self.all_effective_tax_rates.append(effective_tax_rate)
            self._occupancy["{:03d}".format(int(self.income_bin(income)))] += 1

        self.total_collected_taxes += float(net_tax_revenue)

        lump_sum = net_tax_revenue / self.n_agents
        for agent in self.world.agents:
            agent.state["inventory"]["Coin"] += lump_sum
            tax_dict[str(agent.idx)]["lump_sum"] = float(lump_sum)
            self.last_coin[agent.idx] = float(agent.total_endowment("Coin"))

        self.taxes.append(tax_dict)

        # Pre-compute some things that will be useful for generating observations.
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        # Fold this period's tax data into the saez buffer.
        if self.tax_model == "saez":
            self._update_saez_buffer(tax_dict)

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        If using the "model_wrapper" tax model and taxes are enabled, the planner's
        action space includes an action subspace for each of the tax brackets. Each
        such action space has as many actions as there are discretized tax rates.
        """
        # Only the planner takes actions through this component.
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized
                # tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.n_disc_rates)
                    for r in self.bracket_cutoffs
                ]

        # Return 0 (no added actions) if the other conditions aren't met.
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        On the first day of each tax period, update taxes. On the last day, enact them.
        """

        # 1. On the first day of a new tax period: Set up the taxes for this period.
        if self.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.set_new_period_rates_model()

            if self.tax_model == "saez":
                self.compute_and_set_new_period_rates_from_saez_formula()

            # (cache this for faster obs generation)
            self._curr_rates_obs = np.array(self.curr_marginal_rates)

        # 2. On the last day of the tax period: Get $-taxes AND update agent endowments.
        if self.tax_cycle_pos >= self.period:
            self.enact_taxes()
            self.tax_cycle_pos = 0

        else:
            self.taxes.append([])

        # increment timestep.
        self.tax_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the tax period cycle they are, information about the
        last period's incomes, and the current marginal tax rates, including the
        marginal rate that will apply to their next unit of income.

        The planner observes the same type of information, but for all the agents. It
        also sees, for each agent, their marginal tax rate and reported income from
        the previous tax period.
        """
        is_tax_day = float(self.tax_cycle_pos >= self.period)
        is_first_day = float(self.tax_cycle_pos == 1)
        tax_phase = self.tax_cycle_pos / self.period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self._last_income_obs_sorted,
            curr_rates=self._curr_rates_obs,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)

            curr_marginal_rate = self.marginal_rate(
                agent.total_endowment("Coin") - self.last_coin[i]
            )

            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self._last_income_obs_sorted,
                curr_rates=self._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
            )

            obs["p" + k] = dict(
                last_income=self._last_income_obs[i],
                last_marginal_rate=self.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Masks only apply to the planner and if tax_model == "model_wrapper" and taxes
        are enabled.
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.tax_cycle_pos==1 (meaning a new tax period is starting).
        When self.tax_cycle_pos==1, tax actions are masked in order to enforce any
        tax annealing.
        """
        if (
            completions != self._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self._last_completions = int(completions)
            self._annealed_rate_max = annealed_tax_limit(
                completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self._planner_masks is None:
                    planner_masks = {
                        k: annealed_tax_mask(
                            completions,
                            self._annealing_warmup,
                            self._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self._planner_tax_val_dict.items()
                    }
                    self._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset trackers.
        """
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        self.tax_cycle_pos = 1
        self.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        self.taxes = []
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self._planner_masks = None

        if self.tax_model == "saez":
            self.curr_bracket_tax_rates = np.array(self.running_avg_tax_rates)

    def get_metrics(self):
        """
        See base_component.py for detailed description.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self._occupancy.values())))
        for c in self.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.total_collected_taxes)

            # Indices of richest and poorest agents.
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode
                # for the richest and poorest agents.
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

            if self.tax_model == "saez":
                # Include the running estimate of elasticity.
                out["saez/estimated_elasticity"] = self.elas_tm1

        return out

    def get_dense_log(self):
        """
        Log taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a tax period
                ended and taxes were collected. For those timesteps, each entry
                contains the tax schedule, each agent's reported income, tax paid,
                and redistribution received.
                Returns None if taxes are disabled.
        """
        if self.disable_taxes:
            return None
        return self.taxes


@component_registry.add
class SimpleLabor(BaseComponent):
    """
    Allows Agents to select a level of labor, which earns income based on skill.

    Labor is "simple" because this simplifies labor to a choice along a 1D axis. More
    concretely, this component adds 100 labor actions, each representing a choice of
    how many hours to work, e.g. action 50 represents doing 50 hours of work; each
    Agent earns income proportional to the product of its labor amount (representing
    hours worked) and its skill (representing wage), with higher skill and higher labor
    yielding higher income.

    This component is intended to be used with the 'PeriodicBracketTax' component and
    the 'one-step-economy' scenario.

    Args:
        mask_first_step (bool): Defaults to True. If True, masks all non-0 labor
            actions on the first step of the environment. When combined with the
            intended component/scenario, the first env step is used to set taxes
            (via the 'redistribution' component) and the second step is used to
            select labor (via this component).
        payment_max_skill_multiplier (float): When determining the skill level of
            each Agent, sampled skills are clipped to this maximum value.
    """

    name = "SimpleLabor"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        payment_max_skill_multiplier=3,
        pareto_param=4.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # This defines the size of the action space (the max # hours an agent can work).
        self.num_labor_hours = 100  # max 100 hours

        assert isinstance(mask_first_step, bool)
        self.mask_first_step = mask_first_step

        self.is_first_step = True
        self.common_mask_on = {
            agent.idx: np.ones((self.num_labor_hours,)) for agent in self.world.agents
        }
        self.common_mask_off = {
            agent.idx: np.zeros((self.num_labor_hours,)) for agent in self.world.agents
        }

        # Skill distribution
        self.pareto_param = float(pareto_param)
        assert self.pareto_param > 0
        self.payment_max_skill_multiplier = float(payment_max_skill_multiplier)
        pmsm = self.payment_max_skill_multiplier
        num_agents = len(self.world.agents)
        # Generate a batch (1000) of num_agents (sorted/clipped) Pareto samples.
        pareto_samples = np.random.pareto(4, size=(1000, num_agents))
        clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
        sorted_clipped_skills = np.sort(clipped_skills, axis=1)
        # The skill level of the i-th skill-ranked agent is the average of the
        # i-th ranked samples throughout the batch.
        self.skills = sorted_clipped_skills.mean(axis=0)

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"skill": 0, "production": 0}
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True
        for agent in self.world.agents:
            agent.state["skill"] = self.skills[agent.idx]

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.num_labor_hours
        return None

    def generate_masks(self, completions=0):
        if self.is_first_step:
            self.is_first_step = False
            if self.mask_first_step:
                return self.common_mask_off

        return self.common_mask_on

    def component_step(self):

        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0:  # NO-OP.
                # Agent is not interacting with this component.
                continue

            if 1 <= action <= self.num_labor_hours:  # set reopening phase

                hours_worked = action  # NO-OP is 0 hours.
                agent.state["endogenous"]["Labor"] = hours_worked

                payoff = hours_worked * agent.state["skill"]
                agent.state["production"] += payoff
                agent.inventory["Coin"] += payoff

            else:
                # If action > num_labor_hours, this is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "skill": agent.state["skill"] / self.payment_max_skill_multiplier
            }
        return obs_dict
