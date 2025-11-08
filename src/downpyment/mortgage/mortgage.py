from typing import Optional, Literal
from downpyment.amortization import (
    Amortization,
    VanillaAmortization,
    AmortizationParams,
)
from downpyment.mortgage.simulation import MortgageSimulation
from downpyment.interest import Interest
from dataclasses import dataclass
from math import log


@dataclass
class EarlyPayment:
    amount: float
    pay_each: int
    reduce: Literal["quota", "term"] = "quota"


def french_quota(interest_rate, n_steps, loan_amount) -> float:
    r = interest_rate
    n = n_steps
    P = loan_amount
    return P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


class Mortgage:
    def __init__(
        self,
        property_price: float,
        interest: Interest,
        n_steps: int,
        amortization: Optional[Amortization] = None,
        downpayment: float = 0.0,
        step_payment_fn: Optional[callable] = None,
        tax_perc: float = 0.0,
    ):
        self.property_price = property_price
        self.downpayment = downpayment
        self.interest = interest
        self.n_steps = n_steps * interest.scale
        self.amortization = amortization or VanillaAmortization()
        self.step_payment_fn = step_payment_fn or french_quota
        self.tax_perc = tax_perc

    @property
    def initial_mortgage(self) -> float:
        return self.property_price * (1 + self.tax_rate) - self.downpayment

    @property
    def interest_rate(self) -> float:
        return self.interest.rate / self.interest.scale

    @property
    def initial_quota(self) -> float:
        return self.calculate_step_payment(self.initial_mortgage)

    @property
    def tax_rate(self) -> float:
        return self.tax_perc / 100

    def calculate_remaining_steps(
        self, remaining_mortgage: float, step_payment: float, interest_rate: float
    ) -> int:
        numerator = log(step_payment) - log(
            step_payment - interest_rate * remaining_mortgage
        )
        denominator = log(1 + self.interest_rate)
        return numerator / denominator

    def calculate_step_payment(self, remaining_mortgage: float) -> float:
        return self.step_payment_fn(
            interest_rate=self.interest_rate,
            n_steps=self.n_steps,
            loan_amount=remaining_mortgage,
        )

    def simulate(
        self, early_payment: Optional[EarlyPayment] = None
    ) -> MortgageSimulation:
        simulation = MortgageSimulation()
        remaining_mortgage = self.initial_mortgage
        step_payment = self.initial_quota
        remaining_steps = self.n_steps
        step = 0
        while (remaining_steps := remaining_steps - 1) >= 0 and remaining_mortgage > 0:
            step += 1
            if early_payment and step % early_payment.pay_each == 0:
                remaining_mortgage -= early_payment.amount
                if early_payment.reduce == "quota":
                    step_payment = self.calculate_step_payment(remaining_mortgage)
                elif early_payment.reduce == "term":
                    # Keep the same quota, update the number of remaining steps
                    remaining_steps = self.calculate_remaining_steps(
                        remaining_mortgage, step_payment, self.interest_rate
                    )
                else:
                    raise ValueError("Invalid early payment reduction method.")

            amortization_params = AmortizationParams(
                remaining_mortgage=remaining_mortgage,
                step_fee=step_payment,
                step_rate=self.interest_rate,
                step=step,
            )
            ammortization_result = self.amortization(amortization_params)

            simulation.capital.append(ammortization_result.capital)
            simulation.interest.append(ammortization_result.interest)
            remaining_mortgage = ammortization_result.remaining_mortgage
            simulation.remaining.append(max(0, remaining_mortgage))
        return simulation
