from behave import *

@given(u'a dependency for energy_supply is water')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given a dependency for energy_supply is water')

@given(u'that nismod is initialised with a water_supply sector')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given that nismod is initialised with a water_supply sector')

@given(u'a dependency for water_supply is electricity')
def step_impl(context):
    raise NotImplementedError(u'STEP: Given a dependency for water_supply is electricity')

@then(u'a cyclic-dependency error is raised')
def step_impl(context):
    raise NotImplementedError(u'STEP: Then a cyclic-dependency error is raised')
