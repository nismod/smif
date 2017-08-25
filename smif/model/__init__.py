"""
"""


def element_before(element, list_):
    """Return the element before a given element in a list, or None if the
    given element is first or not in the list.
    """
    if element not in list_ or element == list_[0]:
        return None
    else:
        index = list_.index(element)
        return list_[index - 1]


def element_after(element, list_):
    """Return the element after a given element in a list, or None if the
    given element is last or not in the list.
    """
    if element not in list_ or element == list_[-1]:
        return None
    else:
        index = list_.index(element)
        return list_[index + 1]
