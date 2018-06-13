import logging
import os
import shutil

import pkg_resources

LOGGER = logging.getLogger(__name__)


def setup_project_folder(args):
    """Creates folder structure in the target directory

    Parameters
    ----------
    args
    """
    _recursive_overwrite('smif', 'sample_project', args.directory)
    if args.directory == ".":
        dirname = "the current directory"
    else:
        dirname = args.directory
    LOGGER.info("Created sample project in %s", dirname)


def _recursive_overwrite(pkg, src, dest):
    if pkg_resources.resource_isdir(pkg, src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        contents = pkg_resources.resource_listdir(pkg, src)
        for item in contents:
            _recursive_overwrite(pkg,
                                 os.path.join(src, item),
                                 os.path.join(dest, item))
    else:
        filename = pkg_resources.resource_filename(pkg, src)
        shutil.copyfile(filename, dest)
