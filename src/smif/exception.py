"""Holds custom smif exception hierarchy

Exception
  +-- SmifException
        +-- SmifDataError
              +-- SmifDataNotFoundError
              +-- SmifDataExistsError
              +-- SmifDataMismatchError
              +-- SmifDataReadError
        +-- ModelRunError
        +-- ValidationError
"""


class SmifException(Exception):
    """The base class for all errors raised in smif
    """
    pass


class SmifDataError(SmifException):
    """Errors raised by the DataInterface
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


class SmifModelRunError(SmifException):
    """Raise when model run requirements are not satisfied
    """
    pass


class ValidationError(SmifException):
    """Custom exception to use for parsing validation.
    """
    pass
