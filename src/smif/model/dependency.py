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
            diff = ""
            if source.dtype != sink.dtype:
                diff += "dtype(%s!=%s) " % (source.dtype, sink.dtype)
            if source.dims != sink.dims:
                diff += "dims(%s!=%s) " % (source.dims, sink.dims)
            if source.coords != sink.coords:
                diff += "coords do not match "
            if source.unit != sink.unit:
                diff += "unit(%s!=%s) " % (source.unit, sink.unit)
            msg = "Dependencies must connect identical metadata (up to variable name). " + \
                "Connecting %s:%s->%s:%s with mismatched %s"
            raise ValueError(
                msg % (source_model.name, source.name, sink_model.name, sink.name, diff))

        self.source_model = source_model
        self.source = source
        self.sink_model = sink_model
        self.sink = sink

    def __repr__(self):
        return "<Dependency({}, {}, {}, {})>".format(
            self.source_model, self.source, self.sink_model, self.sink)

    def __eq__(self, other):
        return self.source_model == other.source_model \
            and self.source == other.source \
            and self.sink_model == other.sink_model \
            and self.sink == other.sink
