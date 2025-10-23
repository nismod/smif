import logging
import os
import shutil

from importlib import resources

def copy_project_folder(directory):
    """Creates folder structure in the target directory

    Parameters
    ----------
    directory:
        Location where the sample project should be copied to
    """
    _recursive_overwrite("smif", "sample_project", directory)
    if directory == ".":
        dirname = "the current directory"
    else:
        dirname = directory
    logging.info("Created sample project in %s", dirname)


def _recursive_overwrite(pkg: str, src: str, dest: str):
    if resources.files(pkg).joinpath(src).is_dir():
        if not os.path.isdir(dest):
            os.makedirs(dest)
        contents = resources.files(pkg).joinpath(src).iterdir()
        for item in contents:
            _recursive_overwrite(pkg, os.path.join(src, item.name), os.path.join(dest, item.name))
    else:
        filename = resources.files(pkg) / src
        with resources.as_file(filename) as path:
            shutil.copyfile(path, dest)
