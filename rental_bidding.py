import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random


class MarketCondition(Enum):
    """
    Enumeration representing different rental market conditions.

    COOLING: Slower market with fewer competitors, allows below-asking bids
    BALANCED: Normal market conditions with moderate competition
    HOT: Competitive market requiring above-asking bids to succeed
    """
    COOLING = "cooling"
    BALANCED = "balanced"
    HOT = "hot"


class LandlordAction(Enum):
    """
    Enumeration of possible landlord responses to tenant bids.

    ACCEPT_TENANT: Landlord accepts the tenant's offer
    ACCEPT_COMPETITOR: Landlord accepts a competitor's offer
    REQUEST_BEST_FINAL: Landlord requests best and final offers from all parties
    REJECT_ALL: Landlord rejects all offers and relists the property
    """
    ACCEPT_TENANT = "accept_tenant"
    ACCEPT_COMPETITOR = "accept_competitor"
    REQUEST_BEST_FINAL = "request_best_final"
    REJECT_ALL = "reject_all"


# 2024 CMHC Vancouver rental data
VANCOUVER_RENT_DATA = {
    "West End / Stanley Park": {
        "Studio": 1667, "1-Bedroom": 1835, "2-Bedroom": 2609, "3-Bedroom": 3584
    },
    "English Bay": {
        "Studio": 1612, "1-Bedroom": 1970, "2-Bedroom": 2758, "3-Bedroom": 3888
    },
    "Downtown": {
        "Studio": 1783, "1-Bedroom": 2076, "2-Bedroom": 3189, "3-Bedroom": 4814
    },
    "South Granville / Oak": {
        "Studio": 1507, "1-Bedroom": 1816, "2-Bedroom": 2411, "3-Bedroom": 3368
    },
    "Kitsilano / Point Grey": {
        "Studio": 1509, "1-Bedroom": 1870, "2-Bedroom": 2685, "3-Bedroom": 3621
    },
    "Westside / Kerrisdale": {
        "Studio": 1749, "1-Bedroom": 1833, "2-Bedroom": 2862, "3-Bedroom": 4210
    },
    "Marpole": {
        "Studio": 1306, "1-Bedroom": 1475, "2-Bedroom": 1809, "3-Bedroom": None
    },
    "Mount Pleasant / Renfrew Heights": {
        "Studio": 1624, "1-Bedroom": 1732, "2-Bedroom": 2413, "3-Bedroom": 4145
    },
    "East Hastings": {
        "Studio": 1625, "1-Bedroom": 1600, "2-Bedroom": 2402, "3-Bedroom": 2824
    },
    "Southeast Vancouver": {
        "Studio": 1240, "1-Bedroom": 1729, "2-Bedroom": 2120, "3-Bedroom": 2232
    }
}


@dataclass
class PropertyInfo:
    """
    Data structure containing all property-specific information for rental bidding analysis.

    Attributes:
        area: Vancouver neighborhood name (must match VANCOUVER_RENT_DATA keys)
        bedroom_type: Property size category (Studio, 1-Bedroom, 2-Bedroom, 3-Bedroom)
        listing_price: Landlord's advertised monthly rent
        days_on_market: How long the property has been listed (affects landlord desperation)
        max_budget: Tenant's maximum affordable monthly rent
        market_condition: Current rental market heat level
    """
    area: str
    bedroom_type: str
    listing_price: float
    days_on_market: int
    max_budget: float
    market_condition: MarketCondition


@dataclass
class GameState:
    """
    Represents the current state of the rental bidding game at any point in time.

    Attributes:
        round: Current game round (1=initial bids, 2=best&final requested, 3=final decision)
        tenant_bid: Tenant's current bid amount
        competitor_max_bid: Highest competing bid from other applicants
        landlord_action: Landlord's response to bids (None if not yet decided)
        property_info: Static property information
        final_tenant_bid: Tenant's final bid in best&final round (None if not applicable)
    """
    round: int
    tenant_bid: float
    competitor_max_bid: float
    landlord_action: Optional[LandlordAction]
    property_info: PropertyInfo
    final_tenant_bid: Optional[float] = None


class VancouverRentalBiddingGame:
    """
    Main class implementing the rental bidding game using expectiminimax algorithm.

    This class models the strategic interaction between tenants, competitors, and landlords
    in Vancouver's rental market, using real market data and game theory principles.
    """

    def __init__(self,
                 utility_weights: Optional[Dict[str, float]] = None,
                 num_competitors: Optional[int] = None):
        """
        Initialize the rental bidding game with customizable parameters.

        Args:
            utility_weights: Dict of weights for utility calculation components
                - 'base_reward': Positive utility for securing housing (default: 100.0)
                - 'budget_stress': Penalty multiplier for approaching budget limit (default: 2.0)
            num_competitors: Fixed number of competitors (if None, estimated from market conditions)
        """
        self.utility_weights = utility_weights or {
            'base_reward': 100.0,  # Base reward for securing housing
            'budget_stress': 2.0   # Penalty for approaching budget limit
        }
        self.num_competitors = num_competitors
        self.vancouver_data = VANCOUVER_RENT_DATA

    def get_area_average_rent(self, area: str, bedroom_type: str) -> Optional[float]:
        """
        Retrieve average rent for a specific area and bedroom type from Vancouver market data.

        Args:
            area: Vancouver neighborhood name
            bedroom_type: Property size category

        Returns:
            Average rent for the area/type combination, or None if not found
        """
        if area in self.vancouver_data and bedroom_type in self.vancouver_data[area]:
            return self.vancouver_data[area][bedroom_type]
        return None

    def estimate_num_competitors(self, property_info: PropertyInfo) -> int:
        """
        Estimate the number of competing applicants based on market conditions.

        Uses market heat level as the primary factor, as this typically correlates
        with the number of people viewing and applying for properties.

        Args:
            property_info: Property information containing market condition

        Returns:
            Estimated number of competing applicants
        """
        if self.num_competitors is not None:
            return self.num_competitors

        if property_info.market_condition == MarketCondition.COOLING:
            return 1
        elif property_info.market_condition == MarketCondition.BALANCED:
            return 2
        else:
            return 3

    def get_competitor_bid_distribution(self, property_info: PropertyInfo) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation for competitor bid distribution.

        This method uses Vancouver market data to predict how competitors will bid
        based on whether the listing price is above, below, or at market average.

        Args:
            property_info: Property information including area and listing price

        Returns:
            Tuple of (mean_bid, std_deviation) for competitor bidding distribution
        """
        listing_price = property_info.listing_price
        avg_rent = self.get_area_average_rent(
            property_info.area, property_info.bedroom_type)

        # Adjust expectations based on how listing compares to area average
        if avg_rent:
            price_ratio = listing_price / avg_rent

            if price_ratio < 0.95:
                mean = listing_price * 1.08
                std = 0.06 * listing_price
            elif price_ratio > 1.05:
                mean = listing_price * 0.97
                std = 0.03 * listing_price
            else:
                mean = listing_price * 1.02
                std = 0.04 * listing_price
        else:
            # Evaluate based on market condition if no area data available
            if property_info.market_condition == MarketCondition.COOLING:
                mean = 0.95 * listing_price
                std = 0.03 * listing_price
            elif property_info.market_condition == MarketCondition.BALANCED:
                mean = 1.0 * listing_price
                std = 0.05 * listing_price
            else:
                mean = 1.05 * listing_price
                std = 0.07 * listing_price

        return mean, std

    def sample_competitor_bids(self, property_info: PropertyInfo, num_samples: int = 100) -> List[float]:
        """
        Generate sample competitor bids using Monte Carlo simulation.

        For each sample, generates bids for all competitors and returns the maximum
        (since only the highest competing bid matters for landlord decisions).

        Args:
            property_info: Property information for bid distribution calculation
            num_samples: Number of Monte Carlo samples to generate

        Returns:
            List of maximum competitor bids across all samples
        """
        mean, std = self.get_competitor_bid_distribution(property_info)
        num_competitors = self.estimate_num_competitors(property_info)

        samples = []
        for _ in range(num_samples):
            # Sample bids from each competitor
            competitor_bids = np.random.normal(mean, std, num_competitors)
            # Take the maximum bid among all competitors
            samples.append(max(competitor_bids))
        return samples

    def calculate_landlord_desperation(self, days_on_market: int) -> float:
        """
        Calculate landlord's desperation factor based on time property has been listed.

        Uses a sigmoid function that increases landlord willingness to accept lower
        bids as the property sits on market longer, modeling real-world behavior
        where carrying costs and opportunity costs accumulate over time.

        Args:
            days_on_market: Number of days the property has been listed

        Returns:
            Desperation factor between 0 and 1 (0 = not desperate, 1 = very desperate)
        """
        return 1 / (1 + np.exp(-0.1 * (days_on_market - 30)))

    def predict_landlord_action(self, state: GameState) -> Dict[LandlordAction, float]:
        """
        Predict landlord action probabilities based on game state and behavioral modeling.

        Models landlord decision-making considering:
        - Relative bid amounts between tenant and competitors
        - Days on market (desperation factor)
        - Risk aversion vs. profit maximization tradeoffs

        Args:
            state: Current game state with all bid information

        Returns:
            Dictionary mapping each possible landlord action to its probability
        """
        tenant_bid = state.tenant_bid
        competitor_bid = state.competitor_max_bid
        desperation = self.calculate_landlord_desperation(
            state.property_info.days_on_market)

        probs = {}

        # If tenant bid is significantly higher (5%+ advantage)
        if tenant_bid > competitor_bid * 1.05:
            probs[LandlordAction.ACCEPT_TENANT] = 0.7 + desperation * 0.2
            probs[LandlordAction.ACCEPT_COMPETITOR] = 0.05
            probs[LandlordAction.REQUEST_BEST_FINAL] = 0.2 - desperation * 0.15
            probs[LandlordAction.REJECT_ALL] = 0.05 - desperation * 0.05

        # If competitor bid is significantly higher (5%+ advantage)
        elif competitor_bid > tenant_bid * 1.05:
            probs[LandlordAction.ACCEPT_TENANT] = 0.05
            probs[LandlordAction.ACCEPT_COMPETITOR] = 0.7 + desperation * 0.2
            probs[LandlordAction.REQUEST_BEST_FINAL] = 0.2 - desperation * 0.15
            probs[LandlordAction.REJECT_ALL] = 0.05 - desperation * 0.05

        # If bids are close (within 5% of each other)
        else:
            probs[LandlordAction.ACCEPT_TENANT] = 0.15 + desperation * 0.1
            probs[LandlordAction.ACCEPT_COMPETITOR] = 0.15 + desperation * 0.1
            probs[LandlordAction.REQUEST_BEST_FINAL] = 0.6 - desperation * 0.2
            probs[LandlordAction.REJECT_ALL] = 0.1 - desperation * 0.05

        # Normalize probabilities to ensure they sum to 1.0
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

    def calculate_budget_stress(self, bid: float, max_budget: float) -> float:
        """
        Calculate psychological stress/disutility from approaching budget limit.

        Models the increasing anxiety and financial stress tenants experience as their
        bid approaches their maximum affordable rent, using a piecewise function that
        increases exponentially near the budget limit.

        Args:
            bid: Proposed bid amount
            max_budget: Tenant's maximum affordable rent

        Returns:
            Stress factor (0 = no stress, infinity = unaffordable)
        """
        if bid > max_budget:
            return float('inf')

        # Stress function
        budget_usage = bid / max_budget
        if budget_usage < 0.7:
            return 0  # No stress at comfortable budget levels
        elif budget_usage < 0.85:
            # Linear increase in moderate range
            return (budget_usage - 0.7) * 100
        else:
            # Exponential increase near budget limit
            return 15 + np.exp((budget_usage - 0.85) * 20) - 1

    def calculate_utility(self, state: GameState, secured: bool) -> float:
        """
        Calculate tenant utility based on securing housing and associated costs.

        Utility function combines:
        - Base reward for securing housing (positive utility)
        - Budget stress penalty (negative utility increasing with bid amount)
        - Impossibility penalty (negative infinity for unaffordable bids)

        Args:
            state: Current game state containing bid and budget information
            secured: Whether the tenant successfully secured the rental

        Returns:
            Utility score (higher is better, -inf if unaffordable)
        """
        if not secured:
            return -100.0

        w_base = self.utility_weights['base_reward']
        w_budget = self.utility_weights['budget_stress']

        # Determine final price paid (use final bid if available, else regular bid)
        if state.final_tenant_bid is not None:
            final_price = state.final_tenant_bid
        else:
            final_price = state.tenant_bid

        # Check affordability constraint
        if final_price > state.property_info.max_budget:
            return float('-inf')

        # Calculate utility components
        base_reward = w_base
        budget_stress = self.calculate_budget_stress(
            final_price, state.property_info.max_budget)

        utility = base_reward - w_budget * budget_stress

        return utility

    def expectiminimax(self, state: GameState, depth: int, is_tenant_turn: bool,
                       alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[float, float]:
        """
        Expectiminimax algorithm with alpha-beta pruning for rental bidding optimization.

        This algorithm handles the three types of nodes in the game tree:
        1. MAX nodes (tenant's turn): Maximize expected utility
        2. EXPECTATION nodes (landlord/chance): Calculate expected value over probabilistic outcomes
        3. Terminal nodes: Evaluate final utility

        PRUNING LOGIC:
        - Alpha: Best value tenant can guarantee so far (lower bound for tenant)
        - Beta: Upper bound on what expectation nodes will allow
        - Alpha-beta pruning in MAX nodes: stop when best_utility >= beta
        - Expectation pruning: stop evaluating low-probability outcomes when they can't change result significantly

        Args:
            state: Current game state
            depth: Remaining search depth (0 = terminal)
            is_tenant_turn: True for MAX nodes (tenant), False for EXPECTATION nodes (landlord)
            alpha: Best guaranteed utility for tenant so far
            beta: Upper bound constraint from parent expectation node

        Returns:
            Tuple of (best_utility, best_bid_amount)
        """
        # Terminal condition: reached maximum depth or game end
        if depth == 0 or state.round >= 3:
            if state.landlord_action == LandlordAction.ACCEPT_TENANT:
                return self.calculate_utility(state, True), state.tenant_bid
            elif state.landlord_action in [LandlordAction.ACCEPT_COMPETITOR, LandlordAction.REJECT_ALL]:
                return self.calculate_utility(state, False), state.tenant_bid
            else:
                return 0, state.tenant_bid

        if is_tenant_turn:
            # MAX NODE: Tenant chooses bid to maximize expected utility
            if state.round == 1:
                # Initial bidding round - tenant selects optimal initial bid
                best_utility = float('-inf')
                best_bid = state.property_info.listing_price

                # Try different bidding strategies
                # Start with strategies near expected optimal (102%)
                bid_strategies = sorted([0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.10],
                                        key=lambda x: abs(x - 1.02))

                for bid_pct in bid_strategies:
                    bid = state.property_info.listing_price * bid_pct

                    # Skip bids that exceed budget
                    if bid > state.property_info.max_budget:
                        continue

                    new_state = GameState(
                        round=state.round,
                        tenant_bid=bid,
                        competitor_max_bid=state.competitor_max_bid,
                        landlord_action=None,
                        property_info=state.property_info
                    )

                    # Recurse to expectation node (landlord's turn)
                    utility, _ = self.expectiminimax(
                        new_state, depth-1, False, alpha, beta)

                    # Update best choice if this is better
                    if utility > best_utility:
                        best_utility = utility
                        best_bid = bid
                        alpha = max(alpha, best_utility)

                    # PRUNING
                    if best_utility >= beta:
                        break  # Beta cutoff - parent won't choose this branch

                return best_utility, best_bid

            elif state.round == 3:
                # Best and final offer round - tenant decides final bid increment
                if state.landlord_action == LandlordAction.REQUEST_BEST_FINAL:
                    increment_strategies = [1.0, 1.02, 1.04, 1.06]
                    best_utility = float('-inf')
                    best_bid = state.tenant_bid

                    for increment in increment_strategies:
                        new_bid = state.tenant_bid * increment

                        # Check budget constraint
                        if new_bid > state.property_info.max_budget:
                            continue

                        new_state = GameState(
                            round=state.round,
                            tenant_bid=state.tenant_bid,
                            competitor_max_bid=state.competitor_max_bid,
                            landlord_action=state.landlord_action,
                            property_info=state.property_info,
                            final_tenant_bid=new_bid
                        )

                        # Landlord decision for final round
                        # If tenant's final bid is substantially better, tenant wins
                        if new_bid > state.competitor_max_bid * 1.02:
                            utility = self.calculate_utility(new_state, True)
                        else:
                            utility = self.calculate_utility(new_state, False)

                        if utility > best_utility:
                            best_utility = utility
                            best_bid = new_bid
                            alpha = max(alpha, best_utility)

                        # PRUNNING
                        if best_utility >= beta:
                            break  # Beta cutoff

                    return best_utility, best_bid
                else:
                    # Game ended with immediate accept/reject
                    return self.calculate_utility(state,
                                                  state.landlord_action == LandlordAction.ACCEPT_TENANT), state.tenant_bid

        else:
            # EXPECTATION NODE: Landlord makes probabilistic decision
            if state.round == 1:
                # Landlord decides after seeing initial bids
                landlord_probs = self.predict_landlord_action(state)
                expected_utility = 0
                cumulative_prob = 0

                # Sort actions by probability (highest first) for better pruning efficiency
                sorted_actions = sorted(
                    landlord_probs.items(), key=lambda x: x[1], reverse=True)

                for action, prob in sorted_actions:
                    # PRUNING: Skip very low probability outcomes if we've covered most probability mass
                    if cumulative_prob > 0.95 and prob < 0.05:
                        break

                    new_state = GameState(
                        round=2 if action == LandlordAction.REQUEST_BEST_FINAL else 3,
                        tenant_bid=state.tenant_bid,
                        competitor_max_bid=state.competitor_max_bid,
                        landlord_action=action,
                        property_info=state.property_info
                    )

                    if action == LandlordAction.REQUEST_BEST_FINAL:
                        # Continue to best & final round
                        utility, _ = self.expectiminimax(
                            new_state, depth-1, True, alpha, beta)
                    else:
                        # Calculate final utility directly
                        utility = self.calculate_utility(new_state,
                                                         action == LandlordAction.ACCEPT_TENANT)

                    expected_utility += prob * utility
                    cumulative_prob += prob

                    # PRUNING: If remaining probability can't significantly change result
                    remaining_prob = 1 - cumulative_prob
                    if remaining_prob > 0:
                        # Upper bound on remaining utility contribution (assume best case)
                        max_remaining_utility = self.utility_weights['base_reward']
                        max_possible_addition = remaining_prob * max_remaining_utility

                        # Stop if even the best-case scenario can't improve result enough
                        if expected_utility + max_possible_addition < alpha:
                            break

                return expected_utility, state.tenant_bid

        return 0, state.tenant_bid

    def simulate_bidding_scenario(self, property_info: PropertyInfo, tenant_bid: float) -> Dict:
        """
        Simulate a complete bidding scenario from initial bids to final outcome.

        This method provides detailed round-by-round analysis of what happens when
        a tenant submits a specific bid, including:
        - Competitor bid simulation
        - Landlord decision modeling
        - Best & final round dynamics (if applicable)
        - Final outcome determination

        Args:
            property_info: Property details and market conditions
            tenant_bid: Tenant's proposed initial bid

        Returns:
            Dictionary containing:
            - 'rounds': List of round-by-round information
            - 'final_outcome': Description of final result
            - 'secured': Boolean whether tenant got the rental
            - 'final_price': Final rent paid (if secured)
            - 'utility': Calculated utility score
        """
        # Sample competitor bids and convert to float to avoid type issues
        competitor_samples = self.sample_competitor_bids(
            property_info, num_samples=1000)
        competitor_bid = float(np.median(competitor_samples))

        results = {
            'rounds': [],
            'final_outcome': None,
            'secured': False,
            'final_price': None
        }

        # Round 1: Initial bids submitted
        round1_state = GameState(
            round=1,
            tenant_bid=tenant_bid,
            competitor_max_bid=competitor_bid,
            landlord_action=None,
            property_info=property_info
        )

        # Landlord evaluates bids using behavioral model
        landlord_probs = self.predict_landlord_action(round1_state)
        # Most likely action
        landlord_action = max(landlord_probs.items(), key=lambda x: x[1])[0]

        round1_info = {
            'round': 1,
            'tenant_bid': tenant_bid,
            'competitor_bid': competitor_bid,
            'landlord_action': landlord_action.value,
            'landlord_probabilities': {k.value: f"{v*100:.1f}%" for k, v in landlord_probs.items()},
            'bid_difference': tenant_bid - competitor_bid
        }
        results['rounds'].append(round1_info)

        # Process landlord's decision
        if landlord_action == LandlordAction.ACCEPT_TENANT:
            results['final_outcome'] = "Tenant wins immediately!"
            results['secured'] = True
            results['final_price'] = tenant_bid

        elif landlord_action == LandlordAction.ACCEPT_COMPETITOR:
            results['final_outcome'] = "Competitor wins. Tenant loses."
            results['secured'] = False

        elif landlord_action == LandlordAction.REQUEST_BEST_FINAL:
            # Round 2: Best and final offers - matches expectiminimax logic exactly
            increment_strategies = [1.0, 1.02, 1.04, 1.06]
            best_increment = 1.0
            best_utility = float('-inf')

            increment_analysis = []

            for increment in increment_strategies:
                test_bid = tenant_bid * increment

                # Skip if over budget
                if test_bid > property_info.max_budget:
                    increment_analysis.append({
                        'increment_pct': f"{(increment-1)*100:.0f}%",
                        'bid': test_bid,
                        'result': 'Over budget',
                        'utility': float('-inf')
                    })
                    continue

                # Model competitor behavior in best & final
                # Based on market data: competitors typically increase by 1-3%, up to 5% max
                competitor_increase = np.random.normal(
                    1.02, 0.01)  # Mean 2%, std 1%
                competitor_final = competitor_bid * competitor_increase

                # Calculate win probability based on bid difference
                # Strong win (>2% advantage)
                if test_bid > competitor_final * 1.02:
                    win_prob = 0.9
                elif test_bid > competitor_final:  # Marginal win
                    win_prob = 0.6
                else:  # Likely loss
                    win_prob = 0.2

                # Calculate expected utility for this increment
                test_state = GameState(
                    round=2,
                    tenant_bid=tenant_bid,
                    competitor_max_bid=competitor_bid,
                    landlord_action=LandlordAction.REQUEST_BEST_FINAL,
                    property_info=property_info,
                    final_tenant_bid=test_bid
                )

                win_utility = self.calculate_utility(test_state, True)
                lose_utility = self.calculate_utility(test_state, False)
                expected_utility = win_prob * win_utility + \
                    (1 - win_prob) * lose_utility

                increment_analysis.append({
                    'increment_pct': f"{(increment-1)*100:.0f}%",
                    'bid': test_bid,
                    'win_prob': f"{win_prob*100:.0f}%",
                    'utility': expected_utility
                })

                if expected_utility > best_utility:
                    best_utility = expected_utility
                    best_increment = increment

            # Tenant uses optimal increment (matches expectiminimax decision)
            final_tenant_bid = tenant_bid * best_increment

            # Competitor's actual final bid (for the simulation)
            # Mean 2%, std 1%, cap at 5% increase
            competitor_increase = np.random.normal(
                1.02, 0.01)
            final_competitor_bid = competitor_bid * \
                min(competitor_increase, 1.05)

            round2_info = {
                'round': 2,
                'action': 'Best and Final Offers',
                'original_tenant_bid': tenant_bid,
                'increment_analysis': increment_analysis,
                'chosen_increment': f"{(best_increment-1)*100:.0f}%",
                'final_tenant_bid': final_tenant_bid,
                'tenant_increase': f"{(final_tenant_bid/tenant_bid - 1)*100:.1f}%",
                'final_competitor_bid': final_competitor_bid,
                'competitor_increase': f"{(final_competitor_bid/competitor_bid - 1)*100:.1f}%",
                'final_difference': final_tenant_bid - final_competitor_bid
            }
            results['rounds'].append(round2_info)

            # Landlord makes final decision
            if final_tenant_bid > final_competitor_bid:
                results['final_outcome'] = "Tenant wins after best & final!"
                results['secured'] = True
                results['final_price'] = final_tenant_bid
            else:
                results['final_outcome'] = "Competitor wins after best & final."
                results['secured'] = False

        else:
            results['final_outcome'] = "Landlord rejects all offers."
            results['secured'] = False

        # Calculate utility
        final_state = GameState(
            round=2,
            tenant_bid=tenant_bid,
            competitor_max_bid=competitor_bid,
            landlord_action=landlord_action,
            property_info=property_info,
            final_tenant_bid=results.get('final_price')
        )
        results['utility'] = self.calculate_utility(
            final_state, results['secured'])

        return results

    def find_optimal_bid(self, property_info: PropertyInfo) -> Dict:
        """
        Find the optimal initial bid for a property using expectiminimax algorithm.

        This method evaluates different bidding strategies across multiple competitor
        scenarios to find the bid that maximizes expected utility. It uses Monte Carlo
        simulation to handle the uncertainty in competitor behavior.

        Args:
            property_info: Complete property and market information

        Returns:
            Dictionary containing:
            - 'optimal_bid': Best initial bid amount
            - 'optimal_bid_pct': Optimal bid as percentage of listing price
            - 'expected_utility': Expected utility of optimal strategy
            - 'all_strategies': Utility scores for all tested strategies
            - 'budget_usage': Percentage of budget used by optimal bid
            - 'area_average': Average rent for this area/type
            - 'estimated_competitors': Number of competing applicants
            - 'listing_vs_average': How listing price compares to area average
        """
        # Calculate expected utility across competitor bid samples
        competitor_samples = self.sample_competitor_bids(property_info)
        num_competitors = self.estimate_num_competitors(property_info)

        best_overall_utility = float('-inf')
        best_overall_bid = property_info.listing_price
        bid_utilities = {}

        # Try different initial bid strategies (as percentages of listing price)
        bid_strategies = [0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.10]

        for bid_pct in bid_strategies:
            bid = property_info.listing_price * bid_pct

            # Skip strategies that exceed budget
            if bid > property_info.max_budget:
                continue

            # Calculate expected utility across competitor bid samples using Monte Carlo
            total_utility = 0
            for competitor_max in competitor_samples:
                initial_state = GameState(
                    round=1,
                    tenant_bid=bid,
                    competitor_max_bid=float(
                        competitor_max),
                    landlord_action=None,
                    property_info=property_info
                )

                # Use expectiminimax to find expected utility for this bid against this competitor scenario
                utility, _ = self.expectiminimax(
                    initial_state, depth=3, is_tenant_turn=False)
                total_utility += utility

            avg_utility = total_utility / len(competitor_samples)
            bid_utilities[bid_pct] = avg_utility

            # Update best strategy if this one is better
            if avg_utility > best_overall_utility:
                best_overall_utility = avg_utility
                best_overall_bid = bid

        # Get area average for comparison context
        avg_rent = self.get_area_average_rent(
            property_info.area, property_info.bedroom_type)

        return {
            'optimal_bid': best_overall_bid,
            'optimal_bid_pct': best_overall_bid / property_info.listing_price,
            'expected_utility': best_overall_utility,
            'all_strategies': bid_utilities,
            'budget_usage': (best_overall_bid / property_info.max_budget) * 100,
            'area_average': avg_rent,
            'estimated_competitors': num_competitors,
            'listing_vs_average': (property_info.listing_price / avg_rent * 100) if avg_rent else None
        }


if __name__ == "__main__":
    print("=" * 60)
    print("VANCOUVER RENTAL BIDDING STRATEGY ANALYZER")
    print("=" * 60)

    print("\n📍 ENTER PROPERTY DETAILS:")

    # Area selection
    print("\nAvailable Vancouver areas:")
    for i, area in enumerate(VANCOUVER_RENT_DATA.keys(), 1):
        print(f"  {i}. {area}")
    while True:
        try:
            area_num = int(input("\nSelect area number (1-10): "))
            if 1 <= area_num <= 10:
                area = list(VANCOUVER_RENT_DATA.keys())[area_num - 1]
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

    # Bedroom type selection
    print("\nBedroom types:")
    print("  0. Studio")
    print("  1. 1-Bedroom")
    print("  2. 2-Bedroom")
    print("  3. 3-Bedroom+")
    while True:
        try:
            bed_num = int(input("\nSelect bedroom type (0-3): "))
            if bed_num == 0:
                bedroom_type = "Studio"
                break
            elif bed_num == 1:
                bedroom_type = "1-Bedroom"
                break
            elif bed_num == 2:
                bedroom_type = "2-Bedroom"
                break
            elif bed_num == 3:
                bedroom_type = "3-Bedroom"
                break
            else:
                print("Please enter a number between 0 and 3")
        except ValueError:
            print("Please enter a valid number")

    # Area average for reference
    avg_rent = VANCOUVER_RENT_DATA.get(area, {}).get(bedroom_type)
    if avg_rent:
        print(f"\n💡 FYI: Average {bedroom_type} rent in {area}: ${avg_rent}")

    # Listing price input
    while True:
        try:
            listing_price = float(input("\nEnter listing price: $"))
            if listing_price > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    # Days on market input
    while True:
        try:
            days_on_market = int(input("Days on market: "))
            if days_on_market >= 0:
                break
            else:
                print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")

    # Max budget input
    while True:
        try:
            max_budget = float(input("Your maximum budget: $"))
            if max_budget >= listing_price:
                break
            else:
                print(
                    f"Budget should be at least the listing price (${listing_price})")
        except ValueError:
            print("Please enter a valid number")

    # Market condition input
    print("\nMarket conditions:")
    print("  1. COOLING - Low competition, might bid below asking")
    print("  2. BALANCED - Moderate competition, bid around asking")
    print("  3. HOT - High competition, likely need to bid above asking")
    while True:
        try:
            market_num = int(input("\nSelect market condition (1-3): "))
            if market_num == 1:
                market_condition = MarketCondition.COOLING
                break
            elif market_num == 2:
                market_condition = MarketCondition.BALANCED
                break
            elif market_num == 3:
                market_condition = MarketCondition.HOT
                break
            else:
                print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number")

    # Create property info
    property_info = PropertyInfo(
        area=area,
        bedroom_type=bedroom_type,
        listing_price=listing_price,
        days_on_market=days_on_market,
        max_budget=max_budget,
        market_condition=market_condition
    )

    # Create game instance
    game = VancouverRentalBiddingGame()

    # Find optimal initial bid
    print("\n⏳ Analyzing property and calculating optimal strategy...")
    result = game.find_optimal_bid(property_info)

    # Display property analysis
    print(f"\n📍 PROPERTY SUMMARY:")
    print(f"  Location: {property_info.area}")
    print(f"  Type: {property_info.bedroom_type}")
    print(f"  Listing Price: ${property_info.listing_price:.0f}")
    if result['area_average']:
        print(f"  Area Average: ${result['area_average']:.0f}")
        print(
            f"  Price vs Average: {result['listing_vs_average']:.1f}% ({'underpriced' if result['listing_vs_average'] < 100 else 'overpriced'})")
    print(f"  Days on Market: {property_info.days_on_market}")
    print(f"  Market Heat: {property_info.market_condition.value.upper()}")
    print(f"  Expected Competitors: {result['estimated_competitors']}")
    print(f"  Your Budget: ${property_info.max_budget:.0f}")

    # Display strategy analysis
    print(f"\n💡 OPTIMAL STRATEGY:")
    print(f"  Recommended Bid: ${result['optimal_bid']:.0f}")
    print(
        f"  Bid vs Listing: +${(result['optimal_bid'] - property_info.listing_price):.0f} ({result['optimal_bid_pct']*100:.1f}% of listing)")
    print(f"  Budget Usage: {result['budget_usage']:.1f}%")

    # Show what competitors are likely to bid
    competitor_samples = game.sample_competitor_bids(
        property_info, num_samples=1000)
    print(f"\n👥 COMPETITOR ANALYSIS:")
    print(f"  Expected competitor bid range:")
    print(
        f"    - 25th percentile: ${np.percentile(competitor_samples, 25):.0f}")
    print(f"    - Median (50th): ${np.percentile(competitor_samples, 50):.0f}")
    print(
        f"    - 75th percentile: ${np.percentile(competitor_samples, 75):.0f}")

    # Ask if user wants to see simulation
    print(f"\n🎯 Would you like to simulate what happens with the recommended bid?")
    simulate = input("Enter 'y' for yes, any other key to skip: ").lower()

    if simulate == 'y':
        print(
            f"\nSIMULATION: What happens if you bid ${result['optimal_bid']:.0f}?")
        print("-" * 50)

        simulation = game.simulate_bidding_scenario(
            property_info, result['optimal_bid'])

        for round_data in simulation['rounds']:
            if round_data['round'] == 1:
                print(f"\n📝 ROUND 1: Initial Bids Submitted")
                print(f"  Your bid: ${round_data['tenant_bid']:.0f}")
                print(
                    f"  Competitor's best bid: ${round_data['competitor_bid']:.0f}")
                print(
                    f"  Difference: ${round_data['bid_difference']:.0f} {'(you lead)' if round_data['bid_difference'] > 0 else '(competitor leads)'}")
                print(f"\n  Landlord's likely reaction:")
                for action, prob in round_data['landlord_probabilities'].items():
                    print(f"    - {action}: {prob}")
                print(
                    f"  \n  → Landlord decides to: {round_data['landlord_action'].upper()}")

            elif round_data.get('action') == 'Best and Final Offers':
                print(f"\n📝 ROUND 2: Best and Final Offers")
                print(f"\n  Tenant's increment strategy analysis:")
                for analysis in round_data['increment_analysis']:
                    if 'result' in analysis and analysis['result'] == 'Over budget':
                        print(
                            f"    +{analysis['increment_pct']:3} → ${analysis['bid']:.0f} - OVER BUDGET")
                    else:
                        print(
                            f"    +{analysis['increment_pct']:3} → ${analysis['bid']:.0f} - Win prob: {analysis['win_prob']}, Utility: {analysis['utility']:.1f}")

                print(
                    f"\n  → Optimal choice: +{round_data['chosen_increment']} increase")
                print(f"\n  Final bids:")
                print(
                    f"    Your bid: ${round_data['original_tenant_bid']:.0f} → ${round_data['final_tenant_bid']:.0f} ({round_data['tenant_increase']})")
                print(
                    f"    Competitor: ${round_data['final_competitor_bid']:.0f} ({round_data['competitor_increase']})")
                print(
                    f"    Final difference: ${round_data['final_difference']:.0f}")

        print(f"\n🏁 FINAL OUTCOME: {simulation['final_outcome']}")
        if simulation['secured']:
            print(f"  Final rent: ${simulation['final_price']:.0f}/month")
            print(
                f"  Budget usage: {(simulation['final_price']/property_info.max_budget)*100:.1f}%")
        print(f"  Utility score: {simulation['utility']:.1f}")

        # Ask if user wants to see other unchosen strategies
        print(f"\n📊 Would you like to compare different bidding strategies?")
        compare = input("Enter 'y' for yes, any other key to skip: ").lower()

        if compare == 'y':
            print(f"\nSTRATEGY COMPARISON:")
            print("-" * 50)
            strategies = [
                ('Conservative (98% of listing)',
                 property_info.listing_price * 0.98),
                ('At asking (100%)', property_info.listing_price * 1.00),
                ('Slightly above (102%)', property_info.listing_price * 1.02),
                ('Aggressive (105%)', property_info.listing_price * 1.05),
                ('Optimal', result['optimal_bid'])
            ]

            for strategy_name, bid_amount in strategies:
                if bid_amount <= property_info.max_budget:
                    sim = game.simulate_bidding_scenario(
                        property_info, bid_amount)
                    status = "✅ WIN" if sim['secured'] else "❌ LOSE"
                    print(
                        f"  {strategy_name:30} ${bid_amount:4.0f} → {status} (Utility: {sim['utility']:.1f})")
