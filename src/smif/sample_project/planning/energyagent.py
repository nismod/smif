from copy import copy

from smif.decision.decision import RuleBased


class EnergyAgent(RuleBased):
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
        self.model_name = 'energy_supply'

    @staticmethod
    def from_dict(config):
        timesteps = config['timesteps']
        register = config['register']
        return EnergyAgent(timesteps, register)

    def get_decision(self, data_handle):
        budget = self.run_regulator(data_handle)
        decisions = self.run_power_producer(data_handle, budget)
        return decisions

    def run_regulator(self, data_handle):
        """
        data_handle
        """
        budget = 100

        # TODO Should be the iteration previous to the current one?
        if data_handle.current_timestep > data_handle.base_timestep:
            previous_timestep = data_handle.previous_timestep
            iteration = self._max_iteration_by_timestep[previous_timestep]
            output_name = 'cost'
            cost = data_handle.get_results(model_name='energy_demand',
                                           output_name=output_name,
                                           decision_iteration=iteration,
                                           timestep=previous_timestep)
            budget -= sum(cost.as_ndarray())

        self.satisfied = True
        return budget

    def run_power_producer(self, data_handle, budget):
        """
        data_handle
        budget : float
        """
        cheapest_first = []
        for name, item in self.interventions.items():
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
