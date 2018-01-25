"""The decision module handles the three planning levels

Currently, only pre-specified planning is implemented.

The choices made in the three planning levels influence the set of interventions
and assets available within a model run.

The interventions available in a model run are stored in the
:class:`~smif.intervention.InterventionRegister`.

When pre-specified planning are declared, each of the corresponding
interventions in the InterventionRegister are moved to the BuiltInterventionRegister.

Once pre-specified planning is instantiated, the action space for rule-based and
optimisation approaches can be generated from the remaining Interventions in the
InterventionRegister.

"""
__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class Planning:
    """
    Holds the list of planned interventions, where a single planned intervention
    is an intervention with a build date after which it will be included in the
    modelled systems.

    For example, a small pumping station might be built in
    Oxford in 2045::

            {
                'name': 'small_pumping_station',
                'build_date': 2045
            }

    Attributes
    ----------
    planned_interventions : list
        A list of pre-specified planned interventions

    """

    def __init__(self, planned_interventions=None):

        if planned_interventions is not None:
            self.planned_interventions = planned_interventions
        else:
            self.planned_interventions = []

    @property
    def names(self):
        """Returns the set of assets defined in the planned interventions
        """
        return {plan['name'] for plan in self.planned_interventions}

    @property
    def timeperiods(self):
        """Returns the set of build dates defined in the planned interventions
        """
        return {plan['build_date'] for plan in self.planned_interventions}
