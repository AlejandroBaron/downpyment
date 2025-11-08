from typing import Optional
from dataclasses import dataclass, field
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
