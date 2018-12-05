"""Holds custom smif exception hierarchy

Exception
  +-- SmifException
        +-- SmifDataError
              +-- SmifDataNotFoundError
              +-- SmifDataExistsError
              +-- SmifDataMismatchError
              +-- SmifDataReadError
              +-- SmifDataInputError
        +-- SmifModelRunError
        +-- SmifValidationError
"""


class SmifException(Exception):
    """The base class for all errors raised in smif
    """
    pass


class SmifDataError(SmifException):
    """Errors raised by the Store
    """
    pass


class SmifDataNotFoundError(SmifDataError):
    """Raise when some data is not found
    """
    pass


class SmifDataExistsError(SmifDataError):
    """Raise when some data is found unexpectedly
    """
    pass


class SmifDataMismatchError(SmifDataError):
    """Raise when some data doesn't match the context

    E.g. when updating an object by id, the updated object's id must match
    the id provided separately.
    """
    pass


class SmifDataReadError(SmifDataError):
    """Raise when unable to read data

    E.g. unable to handle file type or connect to database
    """
    pass


class SmifDataInputError(SmifDataError):
    """Raise when unable to write data because it does not meet specification
    and can be addressed to a specific (user-interface) input field

    E.g.
    - component: description
    - unable to write a description shorter than 5 characters
    - We require a description so you can identify your system-of-systems
      configuration throughout your project.
    """
    def __init__(self, component, error, message):
        self.component = component
        self.error = error
        self.message = message


class SmifModelRunError(SmifException):
    """Raise when model run requirements are not satisfied
    """
    pass


class SmifValidationError(SmifException):
    """Custom exception to use for parsing validation.
    """
    pass


class SmifTimestepResolutionError(SmifException):
    """Raise when timestep cannot be resolved
    """
    pass
