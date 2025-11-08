# Downpyment


Just a tiny lib to help with mortgage analysis and visualization in Python.

# Installation

You can install the package via pip:

```bash
pip install downpyment
```

# Usage
The central class is the mortgage one

```python
from downpyment.mortgage import Mortgage, Interest, YEARLY_INTEREST_SCALE, Investment
from downpyment.reporting import MortgageReport


mortgage = Mortgage(
    property_price=450_000,
    interest=Interest(rate=1.85, scale=YEARLY_INTEREST_SCALE, perc=True),
    n_steps=30,
    downpayment=0,
    tax_perc=8,
)
mortgage.simulate()
```

The main goal is to generate a report as follows: 
```python
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


MortgageReport(mortgage).report(ep_params=ep_params, inflation_p=2.0, investment=investment)
```

This will generate a report like this one:

# Mortgage simulation with early payments
![Mortgage simulation](mortgage_simulation.png)
![Mortgage simulation with inflation](mortgage_simulation_inflation.png)

# Mortgage vs investment curves
![Mortgage vs investment](mortgage_vs_investment.png)
![Mortgage vs investment with inflation](mortgage_vs_investment_inflation.png)


Check the `scripts/simulate.py` file for a complete example.
