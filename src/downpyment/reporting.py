from downpyment.mortgage.mortgage import (
    Mortgage,
    EarlyPayment,
    MortgageSimulation,
    Investment,
)
from matplotlib import pyplot as plt
from typing import Optional
import numpy as np
from pathlib import Path


class MortgageReport:
    def __init__(self, mortgage: Mortgage):
        self.mortgage = mortgage

    def ep_plot(
        self,
        simulations: dict[str, MortgageSimulation],
        save_path: str = "mortgage_simulation.png",
        inflation_p: Optional[float] = None,
    ):
        fig, axs = plt.subplots(len(simulations), 1, figsize=(10, 6))
        if len(simulations) > 1:
            axs = axs.flatten()
        else:
            axs = [axs]

        for ax, (ep_strategy, simulation) in zip(axs, simulations.items()):
            simulation.capital_vs_interest_plot(ax=ax, inflation_p=inflation_p)
            ax.set_xlim(0, self.mortgage.n_steps)
            title = f"Total payed in interests: ~{simulation.total_interest(inflation_p=inflation_p)/1000:,.0f}k"
            if inflation_p:
                title = f" (Inflation adjusted @ {inflation_p:.2f}%) " + title
            if ep_strategy:
                title = f"E.P => reduce {ep_strategy}. " + title
            ax.set_title(title)

        mortgage = self.mortgage
        fig.suptitle(
            f"~{mortgage.initial_mortgage/1000:,.0f}k Mortgage, {mortgage.interest_rate*12*100:.2f}% yearly interest, Initial quota: {mortgage.initial_quota:,.2f}"
        )
        plt.tight_layout()
        plt.savefig(save_path)

    def mortgage_vs_investment(
        self,
        simulations: dict[str, MortgageSimulation],
        investment: Investment,
        save_path: str = "mortgage_vs_investment.png",
        inflation_p: Optional[float] = None,
    ):
        def _plot(ax, y, label: str, **kwargs):
            steps = list(range(1, len(y) + 1))
            label += f" (~{y[-1]/1000:,.0f}k)"
            ax.plot(steps, y, label=label, **kwargs)

        invest_interests = investment.simulate(
            self.mortgage.n_steps, inflation_p=inflation_p, only_interest=True
        )
        invest_interests = np.cumsum(invest_interests)
        _, ax = plt.subplots(figsize=(10, 6))
        investment_color = "tab:purple"
        _plot(
            ax,
            invest_interests,
            label="Investment Earnings (just interest)",
            color=investment_color,
        )

        for ep_strategy, simulation in simulations.items():
            mortgage_payments = simulation.cumulative_interest(inflation_p=inflation_p)
            label = "Mortgage Interests"
            if ep_strategy:
                label += f" (E.P reduce {ep_strategy})"
            _plot(
                ax,
                mortgage_payments,
                label=label,
                linestyle="--" if ep_strategy else "-.",
            )

        title = "Mortgage Payments vs Investment Evolution"
        if inflation_p:
            title += f" (Inflation adjusted @ {inflation_p:.2f}%)"
        ax.set_xlabel("Step")
        ax.set_ylabel("Amount")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid()
        plt.tight_layout()
        plt.savefig(save_path)

    def report(
        self,
        save_path: str = "outputs",
        ep_params: Optional[dict] = None,
        inflation_p: Optional[float] = None,
        investment: Optional[Investment] = None,
    ):
        save_path: Path = Path(save_path)
        simulations = {
            None: self.mortgage.simulate(),
        }
        if ep_params:
            for reduce in ["quota", "term"]:
                ep = EarlyPayment(
                    **ep_params,
                    reduce=reduce,
                )
                simulations[reduce] = self.mortgage.simulate(early_payment=ep)

        save_path.mkdir(parents=True, exist_ok=True)
        self.ep_plot(
            save_path=save_path / "mortgage_simulation.png", simulations=simulations
        )
        if inflation_p:
            self.ep_plot(
                save_path=save_path / "mortgage_simulation_inflation.png",
                simulations=simulations,
                inflation_p=inflation_p,
            )
        if investment:
            self.mortgage_vs_investment(
                save_path=save_path / "mortgage_vs_investment.png",
                simulations=simulations,
                inflation_p=None,
                investment=investment,
            )
            if inflation_p:
                self.mortgage_vs_investment(
                    save_path=save_path / "mortgage_vs_investment_inflation.png",
                    simulations=simulations,
                    inflation_p=inflation_p,
                    investment=investment,
                )
