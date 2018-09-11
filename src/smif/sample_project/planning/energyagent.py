from copy import copy

from smif.decision.decision import DecisionModule


class EnergyAgent(DecisionModule):
    """A coupled power-producer/regulator decision algorithm for simulating
    decision making in an energy supply model

    EnergyAgent consists of two actors - a PowerProducer and a Regulator.

    The Regulator reviews the performance of the electricity system in the previous period
    against exogenously defined performance metric thresholds,
    such as emissions and capacity margin, and imposes constraints upon the
    available interventions through:
    - emission taxes
    - portfolio standards (e.g. % mandated to come from a generation type)

    In each timestep the PowerProducer assesses the current capacity shortfall
    against its projection of electricity demand and selects a portfolio of
    interventions using a heuristic based upon LCOE (levelised cost of electricity).

    """
    def __init__(self, timesteps, register):
        super().__init__(timesteps, register)
        self.converged = False
        self.current_timestep_index = 0
        self.current_iteration = 0
        self.model_name = 'energy_supply'

    @staticmethod
    def from_dict(config):
        timesteps = config['timesteps']
        register = config['register']
        return EnergyAgent(timesteps, register)

    def _get_next_decision_iteration(self):
            if self.converged and self.current_timestep_index == len(self.timesteps) - 1:
                return None
            elif self.converged and self.current_timestep_index <= len(self.timesteps):
                self.converged = False
                self.current_timestep_index += 1
                self.current_iteration += 1
                return {self.current_iteration: [self.timesteps[self.current_timestep_index]]}
            else:
                self.current_iteration += 1
                return {self.current_iteration: [self.timesteps[self.current_timestep_index]]}

    def get_decision(self, data_handle):
        budget = self.run_regulator(data_handle)
        decisions = self.run_power_producer(data_handle, budget)
        return decisions

    def run_regulator(self, data_handle):
        """
        data_handle
        """

        budget = 100

        iteration = data_handle.decision_iteration
        if data_handle.current_timestep > data_handle.base_timestep:
            output_name = 'cost'
            cost = data_handle.get_results(output_name,
                                           model_name='energy_demand',
                                           decision_iteration=iteration,
                                           timestep=data_handle.previous_timestep)
            budget -= cost

        return budget

    def run_power_producer(self, data_handle, budget):
        """
        data_handle
        budget : float
        """
        cheapest_first = []
        for name, item in self.register.items():
            cheapest_first.append((name, float(item['capital_cost']['value'])))
        sorted(cheapest_first, key=lambda x: float(x[1]), reverse=True)

        within_budget = []
        remaining_budget = copy(budget)
        for intervention in cheapest_first:
            if intervention[1] <= remaining_budget:
                within_budget.append({'name': intervention[0],
                                      'build_year': data_handle.current_timestep})
                remaining_budget -= intervention[1]

        return within_budget
