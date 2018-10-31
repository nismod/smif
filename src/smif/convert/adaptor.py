"""Adaptor is a subclass of :class:`~smif.model.model.Model`, to be used for converting
data between units or dimensions.

The method to override is `generate_coefficients`, which accepts two
:class:`~smif.metadata.spec.Spec` definitions.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from smif.exception import SmifDataNotFoundError
from smif.model import Model


class Adaptor(Model, metaclass=ABCMeta):
    """Abstract Adaptor, to convert inputs/outputs between other Models
    """
    def simulate(self, data):
        """Convert from input to output based on matching variable names
        """
        for from_spec in self.inputs.values():
            if from_spec.name in self.outputs:
                to_spec = self.outputs[from_spec.name]
                coefficients = self.get_coefficients(data, from_spec, to_spec)
                data_in = data.get_data(from_spec.name)
                data_out = self.convert(data_in, from_spec, to_spec, coefficients)
                data.set_results(to_spec.name, data_out)

    def get_coefficients(self, data_handle, from_spec, to_spec):
        """Read coefficients, or generate and save if necessary

        Parameters
        ----------
        data_handle : smif.data_layer.data_handle.DataHandle
        from_spec : smif.metadata.spec.Spec
        to_spec : smif.metadata.spec.Spec

        Returns
        -------
        numpy.ndarray
        """
        try:
            coefficients = data_handle.read_coefficients(from_spec, to_spec)
        except SmifDataNotFoundError:
            msg = "Generating coefficients for %s to %s"
            self.logger.info(msg, from_spec, to_spec)

            coefficients = self.generate_coefficients(from_spec, to_spec)
            data_handle.write_coefficients(from_spec, to_spec, coefficients)
        return coefficients

    @abstractmethod
    def generate_coefficients(self, from_spec, to_spec):
        """Generate coefficients for a pair of :class:`~smif.metadata.spec.Spec` definitions

        Parameters
        ----------
        from_spec : smif.metadata.spec.Spec
        to_spec : smif.metadata.spec.Spec

        Returns
        -------
        numpy.ndarray
        """
        raise NotImplementedError

    def convert(self, data, from_spec, to_spec, coefficients):
        """Convert a dataset between :class:`~smif.metadata.spec.Spec` definitions

        Parameters
        ----------
        data: numpy.ndarray
        from_spec : smif.metadata.spec.Spec
        to_spec : smif.metadata.spec.Spec
        coefficients : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        self.logger.debug("Converting from %s to %s.", from_spec.name, to_spec.name)

        from_convert_dim, to_convert_dim = self.get_convert_dims(from_spec, to_spec)

        self.logger.debug("Converting from %s:%s to %s:%s", from_spec.name, from_convert_dim,
                          to_spec.name, to_convert_dim)

        axis = from_spec.dims.index(from_convert_dim)

        try:
            converted = self.convert_with_coefficients(data, coefficients, axis)
        except ValueError as ex:
            if coefficients.shape[0] != data.shape[axis]:
                msg = "Coefficients do not match dimension to convert: %s != %s"
                raise ValueError(msg, coefficients.shape[0], data.shape[axis]) from ex
            else:
                raise ex

        self.logger.debug("Converted total from %s to %s", data.sum(), converted.sum())
        return converted

    @staticmethod
    def convert_with_coefficients(data, coefficients, axis):
        """Unchecked conversion, given data, coefficients and axis

        Parameters
        ----------
        data : numpy.ndarray
        coefficients : numpy.ndarray
        axis : integer
            Axis along which to apply conversion coefficients

        Returns
        -------
        numpy.ndarray
        """
        # Effectively a tensor contraction (the generalisation of dot product to multi-
        # dimensional ndarrays, tensors) implemented using the Einstein summation convention,
        # np.einsum, which lets us be explicit which dimensions we sum along.

        # coefficients are 2D, label these 0 and 1
        coefficient_axes = [0, 1]

        # data is nD, label these (2 to n+1) to avoid collisions
        data_axes = list(range(2, 2+data.ndim))
        # except for the axis to convert: label this 0 to match first dim of coefficients
        data_axes[axis] = 0

        # results are also nD, label these (2 to n+1) identically to data_axes
        result_axes = list(range(2, 2+data.ndim))
        # except for the axis to convert: label this 1 to match second dim of coefficients
        result_axes[axis] = 1

        return np.einsum(coefficients, coefficient_axes, data, data_axes, result_axes)

    @staticmethod
    def get_convert_dims(from_spec, to_spec):
        """Get dims for conversion from a pair of :class:`~smif.metadata.spec.Spec`,
        assuming only a single dimension will be converted.

        Parameters
        ----------
        from_spec : smif.metadata.Spec
        to_spec : smif.metadata.Spec

        Returns
        -------
        tuple(str)
        """

        from_convert_dims = set(from_spec.dims) - set(to_spec.dims)
        assert len(from_convert_dims) == 1, "Expected a single dim for conversion"
        from_convert_dim = from_convert_dims.pop()

        to_convert_dims = set(to_spec.dims) - set(from_spec.dims)
        assert len(to_convert_dims) == 1, "Expected a single dim for conversion"
        to_convert_dim = to_convert_dims.pop()

        return from_convert_dim, to_convert_dim
