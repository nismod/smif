"""Implements a composite scenario/sector model/system-of-systems model

Begin by declaring the atomic units (scenarios and sector models) which make up
a the composite system-of-systems model and then add these to the composite.
Declare dependencies by using the ``add_dependency()`` method, passing in a
reference to the source model object, and a pointer to the model output
and sink parameter name for the destination model.

Run the model by calling the ``simulate()`` method, passing in a dictionary
containing data for any free hanging model inputs, not linked through a
dependency. A fully defined SosModel should have no hanging model inputs, and
can therefore be called using ``simulate()`` with no arguments.

Responsibility for passing required data to the contained models lies with the
calling class. This means data is only ever passed one layer down.
This simplifies the interface, and allows as little or as much hiding of data,
dependencies and model inputs as required.

Example
-------
A very simple example with just one scenario:

>>> elec_scenario = ScenarioModel('scenario')
>>> elec_scenario.add_output('demand', 'national', 'annual', 'GWh')
>>> sos_model = SosModel('simple')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.simulate(2010)
{'scenario': {'demand': array([123])}}

A more comprehensive example with one scenario and one scenario model:

>>> elec_scenario = ScenarioModel('scenario')
>>> elec_scenario.add_output('demand', 'national', 'annual', 'GWh')
>>> class EnergyModel(SectorModel):
...   def extract_obj(self):
...     pass
...   def simulate(self, timestep, data):
...     return {self.name: {'cost': data['input'] * 2}}
...
>>> energy_model = EnergyModel('model')
>>> energy_model.add_input('input', 'national', 'annual', 'GWh')
>>> energy_model.add_dependency(elec_scenario, 'demand', 'input', lambda x: x)
>>> sos_model = SosModel('sos')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.add_model(energy_model)
>>> sos_model.simulate(2010)
{'model': {'cost': array([[246]])}, 'scenario': {'demand': array([[123]])}}

"""

# import classes for access like ::
#         from smif.model import Model
from smif.model.model import Model, ModelOperation
from smif.model.sector_model import SectorModel
from smif.model.scenario_model import ScenarioModel


# Define what should be imported as * ::
#         from smif.model import *
__all__ = ['Model', 'ModelOperation', 'SectorModel', 'ScenarioModel']
