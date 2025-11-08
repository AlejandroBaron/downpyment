from typing import Optional
from dataclasses import dataclass
from downpyment.interest import Interest


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
