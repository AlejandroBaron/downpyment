from dataclasses import dataclass
from typing import Literal, Optional

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
