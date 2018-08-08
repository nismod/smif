"""Dependencies represent the flow of data explicitly, with reference to the source model
output and sink model input.

A Dependency is an edge in the model process graph. No processing happens on dependency edges,
only on the Model nodes.
"""


class Dependency(object):
    """Link a model input to a data source

    Arguments
    ---------
    source_model : smif.model.Model
        The source model object
    source : smif.metadata.Spec
        The source output description
    sink_model : smif.model.Model
        The sink model object
    sink : smif.metadata.Spec
        The sink input description
    """
    def __init__(self, source_model, source, sink_model, sink):
        # Insist on identical metadata - conversions must be explicit
        if source != sink:
            raise ValueError("Dependencies must connect inputs and outputs with identical " +
                             "metadata (up to variable name).")

        self.source_model = source_model
        self.source = source
        self.sink_model = sink_model
        self.sink = sink

    def __repr__(self):
        return "<Dependency({}, {}, {})>".format(self.source_model, self.source, self.sink)

    def __eq__(self, other):
        return self.source_model == other.source_model \
            and self.source == other.source \
            and self.sink == other.sink
