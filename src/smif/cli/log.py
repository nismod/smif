import datetime
import logging
import logging.config
from collections import OrderedDict


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
        columns = [30, 80, 12]
        column = "{:" + str(columns[0]) + "s}" + \
                 "{:" + str(columns[1]) + "s}" + \
                 "{:" + str(columns[2]) + "s}"

        level_width = max([profile['level'] for profile
                          in logging.Logger._profile.values()]) * 2
        total_width = sum(columns) + level_width

        # header
        summary.append(("{:*^" + str(total_width) + "s}").format(" Modelrun time profile "))
        summary.append(column.format('Function', 'Operation', 'Duration [hh:mm:ss]'))
        summary.append("*"*total_width)

        # body
        for profile in logging.Logger._profile.keys():

            # calculate time diff
            profile_data = logging.Logger._profile[profile]
            diff = profile_data['stop'] - profile_data['start']
            s = diff.total_seconds()
            time_spent = '{:02d}:{:02d}:{:05.2f}'.format(
                int(s // 3600), int(s % 3600 // 60), s % 60)

            # trunctuate long lines
            if len(profile[0]) > columns[0]-2:
                func = profile[0][:columns[0]-3] + '..'
            else:
                func = profile[0]
            if len(profile[1]) > columns[1]-2:
                op = profile[1][:columns[1]-3] + '..'
            else:
                op = profile[1]

            summary.append(profile_data['level']*'| ' + column.format(
                func, op, time_spent))

        # footer
        summary.append("*"*total_width)

        for entry in summary:
            self._log(logging.INFO, entry, args)


def setup_logging(loglevel):
    config = {
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
        },
        # disable_existing_loggers defaults to True, which causes problems with class/module
        # -specific loggers, especially in unit tests when this method might be called multiple
        # times
        'disable_existing_loggers': False
    }

    if loglevel is None:
        config['root']['level'] = logging.WARNING
    elif loglevel == 1:
        config['root']['level'] = logging.INFO
    else:
        config['root']['level'] = logging.DEBUG

    logging.config.dictConfig(config)
    logging.debug('Debug logging enabled.')


logging.Logger.profiling_start = profiling_start
logging.Logger.profiling_stop = profiling_stop
logging.Logger.summary = summary
logging.Logger._profile = OrderedDict()
