from behave import *

@given(u'that nismod is initialised with an energy_supply sector')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given that nismod is initialised with an energy_supply sector')

@given(u'nismod is initialised with a water_supply sector')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given nismod is initialised with a water_supply sector')

@given(u'both of the sectors cover the same region')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given both of the sectors cover the same region')

@given(u'are run for the same time-period')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given are run for the same time-period')

@when(u'the simulation is performed')
def step_impl(context):
    raise NotImplementedError(u'STEP: When the simulation is performed')

@when(u'raininess is 1')
def step_impl(context):
    raise NotImplementedError(u'STEP: When raininess is 1')

@then(u'the energy_supply sector uses all the water')
def step_impl(context):
    raise NotImplementedError(u'STEP: Then the energy_supply sector uses all the water')

@then(u'the energy_supply sector produces 5 electricity')
def step_impl(context):
    raise NotImplementedError(u'STEP: Then the energy_supply sector produces 5 electricity')

@then(u'the water_supply sector raises a shortage of 10')
def step_impl(context):
    raise NotImplementedError(u'STEP: Then the water_supply sector raises a shortage of 10')
