from src.model import BeliefModel
from src.utils import generate_piecewise_linear_irm, generate_linear_supply

# -----------------------------
# Model Parameters (normal distribution)
# -----------------------------
investment_time = 1 / 12
std_beliefs = 2
mean_beliefs = -0.25
traders_initial_budget = 1.0
std_crypto_returns = 0.182
gamma = 0.05
hc = 0.25
hs = 0.30
irm_crypto = generate_piecewise_linear_irm(0.8, 3.8 / 100, 80 / 100)
irm_stablecoin = generate_piecewise_linear_irm(0.9, 4 / 100, 60 / 100)
rho_c = 0.85
rho_s = 0.90
exogenous_supply_crypto = generate_linear_supply(0, 0)
exogenous_supply_stablecoin = generate_linear_supply(0, 0)
size_pop_traders = 250

#Bimodal Model

USE_BIMODAL = True  
mix_pi = 0.53        
mean1  = -0.2       # pessimists' mean belief
std1   = 0.182        # pessimists' dispersion
mean2  =  0.2      # optimists' mean belief
std2   = 0.182        # optimists' dispersion

# -----------------------------
# Model instantiation
# -----------------------------
model = BeliefModel(
    investment_time=investment_time,
    std_beliefs=std_beliefs,
    mean_beliefs=mean_beliefs,
    traders_initial_budget=traders_initial_budget,
    std_crypto_returns=std_crypto_returns,
    gamma=gamma,
    hc=hc,
    hs=hs,
    irm_crypto=irm_crypto,
    irm_stablecoin=irm_stablecoin,
    rho_c=rho_c,
    rho_s=rho_s,
    exogenous_supply_crypto=exogenous_supply_crypto,
    exogenous_supply_stablecoin=exogenous_supply_stablecoin,
    size_pop_traders=size_pop_traders,

    # ---- new args (normal if USE_BIMODAL=False) ----
    bimodal_beliefs=USE_BIMODAL,
    mix_pi=mix_pi,
    mean1=mean1,
    std1=std1,
    mean2=mean2,
    std2=std2,
)

# -----------------------------
# Solve + report
# -----------------------------
uc_eq, us_eq = model.compute_equilibrium_utilization_ratios()
print("uc_eq =", uc_eq, "us_eq =", us_eq)

uc_market, us_market = model.compute_market_utilization_ratios(uc_eq, us_eq)
print("uc_market =", uc_market, "us_market =", us_market)
