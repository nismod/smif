import numpy as np
from pytest import fixture


@fixture(scope='module')
def year_to_month_coefficients():
    """From one year to 12 months

    (apportions)
    """

    month_lengths = np.array([[31, 28, 31, 30, 31, 31, 30, 30, 31, 31, 30, 31]],
                             dtype=np.float).T
    return month_lengths / 365


@fixture(scope='module')
def month_to_year_coefficients():
    """
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
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])  # autumn

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
def months_csv():
    data = [
        {'id': 'jan', 'start': 'P0M', 'end': 'P1M'},
        {'id': 'feb', 'start': 'P1M', 'end': 'P2M'},
        {'id': 'mar', 'start': 'P2M', 'end': 'P3M'},
        {'id': 'apr', 'start': 'P3M', 'end': 'P4M'},
        {'id': 'may', 'start': 'P4M', 'end': 'P5M'},
        {'id': 'jun', 'start': 'P5M', 'end': 'P6M'},
        {'id': 'jul', 'start': 'P6M', 'end': 'P7M'},
        {'id': 'aug', 'start': 'P7M', 'end': 'P8M'},
        {'id': 'sep', 'start': 'P8M', 'end': 'P9M'},
        {'id': 'oct', 'start': 'P9M', 'end': 'P10M'},
        {'id': 'nov', 'start': 'P10M', 'end': 'P11M'},
        {'id': 'dec', 'start': 'P11M', 'end': 'P12M'}
    ]

    return data


@fixture(scope='function')
def months():
    data = [
        ('jan', [('P0M', 'P1M')]),
        ('feb', [('P1M', 'P2M')]),
        ('mar', [('P2M', 'P3M')]),
        ('apr', [('P3M', 'P4M')]),
        ('may', [('P4M', 'P5M')]),
        ('jun', [('P5M', 'P6M')]),
        ('jul', [('P6M', 'P7M')]),
        ('aug', [('P7M', 'P8M')]),
        ('sep', [('P8M', 'P9M')]),
        ('oct', [('P9M', 'P10M')]),
        ('nov', [('P10M', 'P11M')]),
        ('dec', [('P11M', 'P12M')])
    ]

    return data


@fixture(scope='function')
def remap_months_csv():
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
    data = [{'id': 'cold_month', 'start': 'P0M', 'end': 'P1M'},
            {'id': 'cold_month', 'start': 'P1M', 'end': 'P2M'},
            {'id': 'spring_month', 'start': 'P2M', 'end': 'P3M'},
            {'id': 'spring_month', 'start': 'P3M', 'end': 'P4M'},
            {'id': 'spring_month', 'start': 'P4M', 'end': 'P5M'},
            {'id': 'hot_month', 'start': 'P5M', 'end': 'P6M'},
            {'id': 'hot_month', 'start': 'P6M', 'end': 'P7M'},
            {'id': 'hot_month', 'start': 'P7M', 'end': 'P8M'},
            {'id': 'fall_month', 'start': 'P8M', 'end': 'P9M'},
            {'id': 'fall_month', 'start': 'P9M', 'end': 'P10M'},
            {'id': 'fall_month', 'start': 'P10M', 'end': 'P11M'},
            {'id': 'cold_month', 'start': 'P11M', 'end': 'P12M'}]
    return data


@fixture(scope='function')
def remap_months():
    data = [
        ('cold_month', [('P0M', 'P1M'), ('P1M', 'P2M'), ('P11M', 'P12M')]),
        ('spring_month', [('P2M', 'P3M'), ('P3M', 'P4M'), ('P4M', 'P5M')]),
        ('hot_month', [('P5M', 'P6M'), ('P6M', 'P7M'), ('P7M', 'P8M')]),
        ('fall_month', [('P8M', 'P9M'), ('P9M', 'P10M'), ('P10M', 'P11M')])
        ]
    return data


@fixture(scope='function')
def seasons_csv():
    # NB "winter" is split into two pieces around the year end
    data = [{'id': 'winter', 'start': 'P0M', 'end': 'P2M'},
            {'id': 'spring', 'start': 'P2M', 'end': 'P5M'},
            {'id': 'summer', 'start': 'P5M', 'end': 'P8M'},
            {'id': 'autumn', 'start': 'P8M', 'end': 'P11M'},
            {'id': 'winter', 'start': 'P11M', 'end': 'P12M'}]
    return data


@fixture(scope='function')
def seasons():
    # NB "winter" is split into two pieces around the year end
    data = [('winter', [('P0M', 'P2M'), ('P11M', 'P12M')]),
            ('spring', [('P2M', 'P5M')]),
            ('summer', [('P5M', 'P8M')]),
            ('autumn', [('P8M', 'P11M')])]
    return data


@fixture(scope='function')
def twenty_four_hours_csv():
    data = [
        {'id': '1_0', 'start': 'PT0H', 'end': 'PT1H'},
        {'id': '1_1', 'start': 'PT1H', 'end': 'PT2H'},
        {'id': '1_2', 'start': 'PT2H', 'end': 'PT3H'},
        {'id': '1_3', 'start': 'PT3H', 'end': 'PT4H'},
        {'id': '1_4', 'start': 'PT4H', 'end': 'PT5H'},
        {'id': '1_5', 'start': 'PT5H', 'end': 'PT6H'},
        {'id': '1_6', 'start': 'PT6H', 'end': 'PT7H'},
        {'id': '1_7', 'start': 'PT7H', 'end': 'PT8H'},
        {'id': '1_8', 'start': 'PT8H', 'end': 'PT9H'},
        {'id': '1_9', 'start': 'PT9H', 'end': 'PT10H'},
        {'id': '1_10', 'start': 'PT10H', 'end': 'PT11H'},
        {'id': '1_11', 'start': 'PT11H', 'end': 'PT12H'},
        {'id': '1_12', 'start': 'PT12H', 'end': 'PT13H'},
        {'id': '1_13', 'start': 'PT13H', 'end': 'PT14H'},
        {'id': '1_14', 'start': 'PT14H', 'end': 'PT15H'},
        {'id': '1_15', 'start': 'PT15H', 'end': 'PT16H'},
        {'id': '1_16', 'start': 'PT16H', 'end': 'PT17H'},
        {'id': '1_17', 'start': 'PT17H', 'end': 'PT18H'},
        {'id': '1_18', 'start': 'PT18H', 'end': 'PT19H'},
        {'id': '1_19', 'start': 'PT19H', 'end': 'PT20H'},
        {'id': '1_20', 'start': 'PT20H', 'end': 'PT21H'},
        {'id': '1_21', 'start': 'PT21H', 'end': 'PT22H'},
        {'id': '1_22', 'start': 'PT22H', 'end': 'PT23H'},
        {'id': '1_23', 'start': 'PT23H', 'end': 'PT24H'}
    ]
    return data


@fixture(scope='function')
def twenty_four_hours():

    data = [
            ('1_0', [('PT0H', 'PT1H')]),
            ('1_1', [('PT1H', 'PT2H')]),
            ('1_2', [('PT2H', 'PT3H')]),
            ('1_3', [('PT3H', 'PT4H')]),
            ('1_4', [('PT4H', 'PT5H')]),
            ('1_5', [('PT5H', 'PT6H')]),
            ('1_6', [('PT6H', 'PT7H')]),
            ('1_7', [('PT7H', 'PT8H')]),
            ('1_8', [('PT8H', 'PT9H')]),
            ('1_9', [('PT9H', 'PT10H')]),
            ('1_10', [('PT10H', 'PT11H')]),
            ('1_11', [('PT11H', 'PT12H')]),
            ('1_12', [('PT12H', 'PT13H')]),
            ('1_13', [('PT13H', 'PT14H')]),
            ('1_14', [('PT14H', 'PT15H')]),
            ('1_15', [('PT15H', 'PT16H')]),
            ('1_16', [('PT16H', 'PT17H')]),
            ('1_17', [('PT17H', 'PT18H')]),
            ('1_18', [('PT18H', 'PT19H')]),
            ('1_19', [('PT19H', 'PT20H')]),
            ('1_20', [('PT20H', 'PT21H')]),
            ('1_21', [('PT21H', 'PT22H')]),
            ('1_22', [('PT22H', 'PT23H')]),
            ('1_23', [('PT23H', 'PT24H')])]
    return data


@fixture(scope='function')
def one_day_csv():
    data = [{'id': 'one_day', 'start': 'P0D', 'end': 'P1D'}]
    return data


@fixture(scope='function')
def one_day():
    data = [('one_day', [('P0D', 'P1D')])]
    return data


@fixture(scope='function')
def one_year_csv():
    data = [{'id': 'one_year', 'start': 'P0Y', 'end': 'P1Y'}]
    return data


@fixture(scope='function')
def one_year():
    data = [('one_year', [('P0Y', 'P1Y')])]
    return data


@fixture(scope='function')
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = np.array([[
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
        31,
    ]])
    return data


@fixture(scope='function')
def monthly_data_as_seasons():
    return np.array([[
        31+31+28,
        31+30+31,
        30+31+31,
        30+31+30
    ]], dtype=float)


@fixture(scope='function')
def remap_month_data():
    data = np.array([[
        31+31+28,  # Dec, Jan, Feb
        31+30+31,  # Mar, Apr, May
        30+31+31,  # Jun, Jul, Aug
        30+31+30  # Sep, Oct, Nov
    ]], dtype=float) / 3

    return data


@fixture(scope='function')
def remap_month_data_as_months():
    data = np.array([[
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
    ]])
    return data
