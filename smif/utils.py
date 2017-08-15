"""Common utilities for programming convenience
"""


class Singleton(type):
    """Provide a Singleton metaclass. Any class with this metaclass will have
    only a single instance, returned from the usual constructor.

    Pass Singleton as the metaclass when defining a class:
    >>> class MyClass(metaclass=Singleton):
    ...     pass
    ...
    >>> m1 = MyClass()
    >>> m2 = MyClass()
    >>> assert m1 is m2
    """
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(Singleton, cls).__call__(*args, **kwargs)
            return cls.__instance
