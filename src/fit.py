from model import BeliefModel
from typing import Callable
import numpy as np


class EstimatedValues:
    """
    EstimatedValues gathers parameters that are not calibrated by the GMM technique,
    but estimated on the data and then plugged into the GMM. These parameters are:
    - irm_crypto: The irm curve set by the protocol. Given by the Aave Governance.
    - irm_stablecoin: The irm curve set by the protocol. Given by the Aave Governance.
    - hc: the over-collateralization parameter set by the protocol. Set by the Aave Governance.
    - hs: the over-collateralization parameter set by the protocol. Set by the Aave Governance.
    - rho_c: the reserve factor (part of interests given to lenders) parameter set by the protocol. Set by the Aave Governance.
    - rho_s: the reserve factor (part of interests given to lenders) parameter set by the protocol. Set by the Aave Governance.
    """
    def __init__(
        self,
        irm_crypto: Callable[[float], float],
        irm_stablecoin: Callable[[float], float],
        exogenous_supply_crypto: Callable[[float], float],
        exogenous_supply_stablecoin: Callable[[float], float],
        hc: float,
        hs: float,
        rho_c: float,
        rho_s: float,
    ):
        self.irm_crypto = irm_crypto
        self.irm_stablecoin = irm_stablecoin
        self.exogenous_supply_crypto = exogenous_supply_crypto
        self.exogenous_supply_stablecoin = exogenous_supply_stablecoin
        self.hc = hc
        self.hs = hs
        self.rho_c = rho_c
        self.rho_s = rho_s


class ProblemsMomentsSet:
    """
    ProblemsMomentsSet gathers the moments that are used by the GMM technique to calibrate the model.
    - uc: the crypto utilization ratio.
    - us: the stablecoin utilization ratio.
    - ratio_pure_lenders_crypto: the ratio of pure lenders among the total crypto supply.
    - ratio_pure_lenders_stablecoin: the ratio of pure lenders among the total crypto supply.
    - leverage_quantiles: the quantiles of leverage among traders.
    """

    def __init__(
        self,
        uc: float,
        us: float,
        ratio_pure_lenders_crypto: float,
        ratio_pure_lenders_stablecoin: float,
        leverage_quantiles: np.ndarray[float],
    ):
        if not ((0 <= uc <= 1) and (0 <= us <= 1)):
            return ValueError(f"uc and us must be in [0, 1], got {uc} and {us}")
        if not (
            (0 <= ratio_pure_lenders_crypto <= 1)
            and (0 <= ratio_pure_lenders_stablecoin <= 1)
        ):
            return ValueError(
                f"ratio_pure_lenders_crypto and ratio_pure_lenders_stablecoin must be in [0, 1], got {ratio_pure_lenders_crypto} and {ratio_pure_lenders_stablecoin}"
            )

        self.uc = uc
        self.us = us
        self.ratio_pure_lenders_crypto = ratio_pure_lenders_crypto
        self.ratio_pure_lenders_stablecoin = ratio_pure_lenders_stablecoin
        self.leverage_quantiles = leverage_quantiles


class ProblemHyperParametersSet:
    def __init__(
        self,
        gamma: float,
        mean_beliefs: float,
        std_beliefs: float,
        traders_initial_budget: float,
        std_crypto_returns: float,
    ):
        if gamma <= 0:
            return ValueError(f"gamma must be positive, got {gamma}")
        if std_beliefs <= 0:
            return ValueError(f"std_beliefs must be positive, got {std_beliefs}")
        if traders_initial_budget <= 0:
            return ValueError(
                f"traders_initial_budget must be positive, got {traders_initial_budget}"
            )
        if std_crypto_returns <= 0:
            return ValueError(
                f"std_crypto_returns must be positive, got {std_crypto_returns}"
            )

        self.gamma = gamma
        self.mean_beliefs = mean_beliefs
        self.std_beliefs = std_beliefs
        self.traders_initial_budget = traders_initial_budget
        self.std_crypto_returns = std_crypto_returns

    def get_moments(self, estimated_values: EstimatedValues) -> ProblemsMomentsSet:
        """
        Docstring pour get_moments
        
        :param estimated_values: Set of Pre-estimated parameters
        :type estimated_values: EstimatedValues
        :return: The set of moments of interest for the GMM.
        :rtype: ProblemsMomentsSet
        """
        model = BeliefModel(
            investment_time=1 / 12,
            mean_beliefs=self.mean_beliefs,
            std_beliefs=self.std_beliefs,
            traders_initial_budget=self.traders_initial_budget,
            std_crypto_returns=self.std_crypto_returns,
            gamma=self.gamma,
            hc=estimated_values.hc,
            hs=estimated_values.hs,
            irm_crypto=estimated_values.irm_crypto,
            irm_stablecoin=estimated_values.irm_stablecoin,
            rho_c=estimated_values.rho_c,
            rho_s=estimated_values.rho_s,
            exogenous_supply_crypto=estimated_values.exogenous_supply_crypto,
            exogenous_supply_stablecoin=estimated_values.exogenous_supply_stablecoin,
            size_pop_traders=250,
        )
        uc, us = model.compute_equilibrium_utilization_ratios()
        uc_market, us_market = model.compute_market_utilization_ratios(uc, us)
        # Safety check: the obtained utilization ratios are indeed equilibrium.
        assert (abs(uc_market - uc) < 1e-3) and (abs(us_market - us) < 1e-3), (
            f"model is not at equilibrium, deltas are delta_c = {abs(uc_market - uc)} and delta_s = {abs(us_market - us)}"
        )
        ratio_pure_lenders_crypto = model.lending_crypto / (
            model.lending_crypto + model.collat_crypto
        )
        ratio_pure_lenders_stablecoin = model.lending_stablecoin / (
            model.lending_stablecoin + model.collat_stablecoin
        )
        values_leverage = [x.wc for x in model.traders_population]
        weights_leverage = [x.weight for x in model.traders_population]
        return ProblemsMomentsSet(
            uc=uc,
            us=us,
            ratio_pure_lenders_crypto=ratio_pure_lenders_crypto,
            ratio_pure_lenders_stablecoin=ratio_pure_lenders_stablecoin,
            leverage_quantiles=_weighted_deciles(
                values=values_leverage, weights=weights_leverage
            ),
        )


def _weighted_deciles(values, weights):
    """
    Compute 10 weighted quantiles (deciles) from values and weights.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights associated with the values.

    Returns
    -------
    np.ndarray
        Array of length 10 containing the weighted quantiles
        at probabilities 0.1, 0.2, ..., 1.0.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]

    cumulative_weights = np.cumsum(weights_sorted)
    total_weight = cumulative_weights[-1]

    if total_weight == 0:
        raise ValueError("Sum of weights must be positive")

    cumulative_distribution = cumulative_weights / total_weight
    quantile_probs = np.linspace(0.1, 1.0, 10)
    deciles = np.interp(quantile_probs, cumulative_distribution, values_sorted)
    return deciles
