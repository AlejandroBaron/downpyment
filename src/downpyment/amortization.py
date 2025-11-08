from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AmortizationParams:
    """Parameters for an amortization step
    Args:
        remaining_mortgage: remaining mortgage at this step.
        step_fee: total fee to be paid at this step.
        step: current step number.
    """

    remaining_mortgage: float
    step_fee: float
    step_rate: float
    step: int


@dataclass
class AmortizationResult:
    """Result of an amortization step

    Args:
        interest: interest paid in this step.
        capital: capital paid in this step.
    """

    interest: float
    capital: float
    remaining_mortgage: float = 0.0


class Amortization(ABC):
    @abstractmethod
    def __call__(self, params: AmortizationParams) -> AmortizationResult:
        """
        Calculate the interest and capital for the given step.

        Args:
            params: args for the amortization calculation.
        """


class VanillaAmortization(Amortization):
    def __call__(self, params: AmortizationParams) -> AmortizationResult:
        I_k = params.remaining_mortgage * params.step_rate
        C_k = params.step_fee - I_k
        return AmortizationResult(
            interest=I_k,
            capital=C_k,
            remaining_mortgage=params.remaining_mortgage - C_k,
        )
