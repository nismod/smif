"""Conversion data fixtures
"""
import numpy as np
from pytest import fixture


@fixture(scope='module')
def year_to_month_coefficients():
    """From one year to 12 months

    (apportions)
    """
    return np.array([[31, 28, 31, 30, 31, 31, 30, 30, 31, 31, 30, 31]], dtype=np.float).T / 365


@fixture(scope='module')
def month_to_year_coefficients():
    """From 12 months to one year
    """
    return np.ones((1, 12), dtype=np.float)


@fixture(scope='module')
def month_to_season_coefficients():
    """
    12 months to four seasons (winter is December, January, Feb)

    Sum value for each month into season
    """
    coef = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # winter
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # spring
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # summer
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  # autumn
    ])
    return coef.T


@fixture(scope='module')
def season_to_month_coefficients():
    """
    12 months to four seasons (winter is December, January, Feb)

    To convert from seasons to months, find the proportion of each season that
    corresponds to the relevant month.

    E.g. winter to january is (duration of Jan / total duration of winter)
    """
    coef = np.array(
        # winter
        #     spring
        #        summer
        #           autumn
        [[31, 0, 0, 0],  # January
         [28, 0, 0, 0],  # Feb
         [0, 31, 0, 0],  # March
         [0, 30, 0, 0],  # April
         [0, 31, 0, 0],  # May
         [0, 0, 30, 0],  # June
         [0, 0, 31, 0],  # July
         [0, 0, 31, 0],  # August
         [0, 0, 0, 30],  # September
         [0, 0, 0, 31],  # October
         [0, 0, 0, 30],  # November
         [31, 0, 0, 0]]   # December
    )

    days_in_seasons = np.array([
        31+31+28,  # winter
        31+30+31,  # spring
        30+31+31,  # summer
        30+31+30  # autumn
    ], dtype=float)

    return np.transpose(coef / days_in_seasons)


@fixture(scope='function')
def months():
    data = [
        {'name': 'jan', 'interval': [['P0M', 'P1M']]},
        {'name': 'feb', 'interval': [['P1M', 'P2M']]},
        {'name': 'mar', 'interval': [['P2M', 'P3M']]},
        {'name': 'apr', 'interval': [['P3M', 'P4M']]},
        {'name': 'may', 'interval': [['P4M', 'P5M']]},
        {'name': 'jun', 'interval': [['P5M', 'P6M']]},
        {'name': 'jul', 'interval': [['P6M', 'P7M']]},
        {'name': 'aug', 'interval': [['P7M', 'P8M']]},
        {'name': 'sep', 'interval': [['P8M', 'P9M']]},
        {'name': 'oct', 'interval': [['P9M', 'P10M']]},
        {'name': 'nov', 'interval': [['P10M', 'P11M']]},
        {'name': 'dec', 'interval': [['P11M', 'P12M']]},
    ]
    return data


@fixture
def remap_months():
    """Remapping four representative months to months across the year

    In this case we have a model which represents the seasons through
    the year using one month for each season. We then map the four
    model seasons 1, 2, 3 & 4 onto the months throughout the year that
    they represent.

    The data will be presented to the model using the four time intervals,
    1, 2, 3 & 4. When converting to hours, the data will be replicated over
    the year.  When converting from hours to the model time intervals,
    data will be averaged and aggregated.

    """
    data = [
        {'name': 'cold_month', 'interval': [['P0M', 'P1M'], ['P1M', 'P2M'], ['P11M', 'P12M']]},
        {'name': 'spring_month', 'interval': [['P2M', 'P3M'], ['P3M', 'P4M'], ['P4M', 'P5M']]},
        {'name': 'hot_month', 'interval': [['P5M', 'P6M'], ['P6M', 'P7M'], ['P7M', 'P8M']]},
        {'name': 'fall_month', 'interval': [['P8M', 'P9M'], ['P9M', 'P10M'], ['P10M', 'P11M']]}
    ]
    return data


@fixture
def seasons():
    # NB "winter" is split into two pieces around the year end
    data = [
        {'name': 'winter', 'interval': [['P0M', 'P2M'], ['P11M', 'P12M']]},
        {'name': 'spring', 'interval': [['P2M', 'P5M']]},
        {'name': 'summer', 'interval': [['P5M', 'P8M']]},
        {'name': 'autumn', 'interval': [['P8M', 'P11M']]},
    ]
    return data


@fixture(scope='function')
def twenty_four_hours():
    data = [
        {'name': '1_0', 'interval': [['PT0H', 'PT1H']]},
        {'name': '1_1', 'interval': [['PT1H', 'PT2H']]},
        {'name': '1_2', 'interval': [['PT2H', 'PT3H']]},
        {'name': '1_3', 'interval': [['PT3H', 'PT4H']]},
        {'name': '1_4', 'interval': [['PT4H', 'PT5H']]},
        {'name': '1_5', 'interval': [['PT5H', 'PT6H']]},
        {'name': '1_6', 'interval': [['PT6H', 'PT7H']]},
        {'name': '1_7', 'interval': [['PT7H', 'PT8H']]},
        {'name': '1_8', 'interval': [['PT8H', 'PT9H']]},
        {'name': '1_9', 'interval': [['PT9H', 'PT10H']]},
        {'name': '1_10', 'interval': [['PT10H', 'PT11H']]},
        {'name': '1_11', 'interval': [['PT11H', 'PT12H']]},
        {'name': '1_12', 'interval': [['PT12H', 'PT13H']]},
        {'name': '1_13', 'interval': [['PT13H', 'PT14H']]},
        {'name': '1_14', 'interval': [['PT14H', 'PT15H']]},
        {'name': '1_15', 'interval': [['PT15H', 'PT16H']]},
        {'name': '1_16', 'interval': [['PT16H', 'PT17H']]},
        {'name': '1_17', 'interval': [['PT17H', 'PT18H']]},
        {'name': '1_18', 'interval': [['PT18H', 'PT19H']]},
        {'name': '1_19', 'interval': [['PT19H', 'PT20H']]},
        {'name': '1_20', 'interval': [['PT20H', 'PT21H']]},
        {'name': '1_21', 'interval': [['PT21H', 'PT22H']]},
        {'name': '1_22', 'interval': [['PT22H', 'PT23H']]},
        {'name': '1_23', 'interval': [['PT23H', 'PT24H']]},
    ]
    return data


@fixture(scope='function')
def one_day():
    data = [
        {'name': 'one_day', 'interval': [['P0D', 'P1D']]},
    ]
    return data


@fixture(scope='function')
def one_year():
    data = [
        {'name': 'one_year', 'interval': [['P0Y', 'P1Y']]},
    ]
    return data


@fixture(scope='function')
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = np.array([
        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
    ])
    return data


@fixture(scope='function')
def monthly_data_as_seasons():
    return np.array([
        31+31+28,
        31+30+31,
        30+31+31,
        30+31+30
    ], dtype=float)


@fixture(scope='function')
def remap_month_data():
    data = np.array([
        31+31+28,  # Dec, Jan, Feb
        31+30+31,  # Mar, Apr, May
        30+31+31,  # Jun, Jul, Aug
        30+31+30  # Sep, Oct, Nov
    ], dtype=float) / 3

    return data


@fixture(scope='function')
def remap_month_data_as_months():
    data = np.array([
        30.666666666,
        29.666666666,
        29.666666666,
        29.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666
    ])
    return data


@fixture(scope='function')
def regions_rect():
    """Return single region covering 2x1 area::

        |```````|
        |   0   |
        |.......|

    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'zero'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 2], [1, 0]]]
            }
        }
    ]


@fixture(scope='function')
def regions_half_squares():
    """Return two adjacent square regions::

        |```|```|
        | A | B |
        |...|...|

    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'a'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'b'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 1], [0, 2], [1, 2], [1, 1]]]
            }
        }
    ]


@fixture(scope='function')
def regions():
    """Return data structure for test regions/shapes
    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'unit'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'half'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 0.5], [1, 0.5], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'two'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 2], [1, 0]]]
            }
        }
    ]


@fixture(scope='function')
def regions_single_half_square():
    """Return single half-size square region::

        |```|
        | A |
        |...|

    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'a'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        }
    ]


@fixture(scope='function')
def regions_half_triangles():
    """Return regions split diagonally::

        |``````/|
        | 0 / 1 |
        |/......|

    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'zero'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'one'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 2], [1, 2], [1, 0]]]
            }
        }
    ]
