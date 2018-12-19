import datetime
import logging
import logging.config
import re
import sys
from collections import OrderedDict

LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
        },
        'message': {
            'format': '\033[1;34m%(levelname)-8s\033[0m %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'smif.log',
            'mode': 'a',
            'encoding': 'utf-8'
        },
        'stream': {
            'class': 'logging.StreamHandler',
            'formatter': 'message',
            'level': 'DEBUG'
        }
    },
    'root': {
        'handlers': ['file', 'stream'],
        'level': 'DEBUG'
    }
}


# Make profiling methods available through the logger
def profiling_start(self, operation, key):
    time_placeholder = datetime.time(0, 0)
    level = sum(log[1]['stop'] == time_placeholder for log in logging.Logger._profile.items())
    logging.Logger._profile[(operation, key)] = {
        'start': datetime.datetime.now(),
        'stop': time_placeholder,
        'level': level
    }


def profiling_stop(self, operation, key):
    logging.Logger._profile[(operation, key)]['stop'] = datetime.datetime.now()


def summary(self, *args, **kws):
    if self.isEnabledFor(logging.INFO):
        summary = []
        summary.append("*"*150)
        for profile in logging.Logger._profile.keys():
            profile_data = logging.Logger._profile[profile]
            diff = profile_data['stop'] - profile_data['start']
            s = diff.total_seconds()
            time_spent = '{:02d}:{:02d}:{:02d}'.format(
                int(s // 3600), int(s % 3600 // 60), int(s % 60))
            summary.append(profile_data['level']*'| ' + "{:20s} {:80s} {:50s}".format(
                profile[0], profile[1], time_spent))
        summary.append("*"*150)

        for entry in summary:
            self._log(logging.INFO, entry, args)


logging.Logger.profiling_start = profiling_start
logging.Logger.profiling_stop = profiling_stop
logging.Logger.summary = summary
logging.Logger._profile = OrderedDict()

# Configure logging once, outside of any dependency on argparse
VERBOSITY = None
if '--verbose' in sys.argv:
    VERBOSITY = sys.argv.count('--verbose')
else:
    for arg in sys.argv:
        if re.match(r'\A-v+\Z', arg):
            VERBOSITY = len(arg) - 1
            break

if VERBOSITY is None:
    LOGGING_CONFIG['root']['level'] = logging.WARNING
elif VERBOSITY == 1:
    LOGGING_CONFIG['root']['level'] = logging.INFO
else:
    LOGGING_CONFIG['root']['level'] = logging.DEBUG

logging.config.dictConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger(__name__)
LOGGER.debug('Debug logging enabled.')
