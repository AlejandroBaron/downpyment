from typing import Optional, Literal
from downpyment.amortization import (
    Amortization,
    VanillaAmortization,
    AmortizationParams,
)
from dataclasses import dataclass, field
from math import log
import numpy as np


@dataclass
class MortgageSimulation:
    capital: list[float] = field(default_factory=list)
    interest: list[float] = field(default_factory=list)
    remaining: list[float] = field(default_factory=list)

    def iter(self):
        return zip(self.interest, self.capital)

    @staticmethod
    def inflation_adjust(v, p, month):
        return v / (1 + p) ** month

    def total_interest(self, inflation_p: Optional[float] = None) -> float:
        if inflation_p:
            inflation_r = inflation_p / 100 / 12
            return sum(
                self.inflation_adjust(v, inflation_r, month)
                for month, v in enumerate(self.interest)
            )
        return sum(self.interest)

    def total_capital(self, inflation_p: Optional[float] = None) -> float:
        if inflation_p:
            inflation_r = inflation_p / 100 / 12
            return sum(
                self.inflation_adjust(v, inflation_r, month)
                for month, v in enumerate(self.capital)
            )
        return sum(self.capital)

    def payments(self, inflation_p: Optional[float] = None) -> float:
        if inflation_p:
            inflation_r = inflation_p / 100 / 12
            return [
                self.inflation_adjust(c + i, inflation_r, month)
                for month, (i, c) in enumerate(self.iter())
            ]
        return [c + i for i, c in self.iter()]

    def cumulative_interest(self, inflation_p: Optional[float] = None) -> list[float]:
        interests = self.interest
        if inflation_p:
            inflation_r = inflation_p / 100 / 12
            interests = [
                self.inflation_adjust(v, inflation_r, month)
                for month, v in enumerate(self.interest)
            ]
        return np.cumsum(interests).tolist()

    def cumulative_payments(self, inflation_p: Optional[float] = None) -> list[float]:
        payments = self.payments(inflation_p=inflation_p)
        return np.cumsum(payments).tolist()

    def capital_vs_interest_plot(self, ax, inflation_p: Optional[float] = None):
        ax = ax
        steps = list(range(1, len(self.interest) + 1))
        capital, interest = self.capital, self.interest
        if inflation_p:
            inflation_r = inflation_p / 100 / 12
            capital = [
                self.inflation_adjust(v, inflation_r, month)
                for month, v in enumerate(self.capital)
            ]
            interest = [
                self.inflation_adjust(v, inflation_r, month)
                for month, v in enumerate(self.interest)
            ]
        ax.stackplot(steps, [capital, interest], labels=["Capital", "Interests"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Amount")
        ax.legend()
        ax.grid()


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


MONTHLY_INTEREST_SCALE = 1
YEARLY_INTEREST_SCALE = 12


@dataclass
class Interest:
    rate: float
    scale: Literal[1, 12] = YEARLY_INTEREST_SCALE
    perc: Optional[bool] = True

    def __post_init__(self):
        if self.perc:
            self.rate /= 100
            self.perc = False


@dataclass
class Investment:
    initial_amount: float
    step_contribution: float
    interest: Interest
    tax_perc: float = 0.0

    def simulate(
        self,
        n_steps: int,
        inflation_p: Optional[float] = None,
        only_interest: bool = False,
    ) -> list[float]:
        values = []
        current_value = self.initial_amount
        step_interest = (1 + self.interest.rate) ** (1 / self.interest.scale)
        step_inflation = (
            (1 + inflation_p / 100) ** (1 / self.interest.scale)
            if inflation_p
            else None
        )

        for _ in range(n_steps):
            # Interest earned on current value
            interest_earned = current_value * (step_interest - 1)

            # Add interest and contribution
            current_value += interest_earned + self.step_contribution

            # Apply inflation *this step* (reduce real value)
            if step_inflation:
                current_value /= step_inflation

            # Record either total or just interest portion
            values.append(interest_earned if only_interest else current_value)

        tax_r = self.tax_perc / 100
        return [v * (1 - tax_r) for v in values]


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
