# %%
from downpyment.mortgage import Mortgage
from downpyment.interest import Interest, YEARLY_INTEREST_SCALE
from downpyment.investment import Investment
from downpyment.reporting import MortgageReport


mortgage = Mortgage(
    property_price=300_000,
    interest=Interest(rate=1.85, scale=YEARLY_INTEREST_SCALE, perc=True),
    n_steps=30,
    downpayment=30_000,
    tax_perc=8,
)

ep_params = {
    "amount": 5_000,
    "pay_each": 12,
}
investment = Investment(
    initial_amount=120_000,
    step_contribution=ep_params["amount"] / ep_params["pay_each"],
    interest=Interest(rate=5.0, scale=YEARLY_INTEREST_SCALE, perc=True),
    tax_perc=20,
)


MortgageReport(mortgage).report(
    ep_params=ep_params, inflation_p=2.0, investment=investment
)


# %%
