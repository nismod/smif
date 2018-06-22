import logging
import os
import shutil

import pkg_resources

LOGGER = logging.getLogger(__name__)


def copy_project_folder(directory):
    """Creates folder structure in the target directory

    Parameters
    ----------
    directory:
        Location where the sample project should be copied to
    """
    _recursive_overwrite('smif', 'sample_project', directory)
    if directory == ".":
        dirname = "the current directory"
    else:
        dirname = directory
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
