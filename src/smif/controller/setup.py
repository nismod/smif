import logging
import os
from importlib import resources as importlib_resources


def copy_project_folder(directory):
    """Creates folder structure in the target directory

    Parameters
    ----------
    directory:
        Location where the sample project should be copied
    """
    root = importlib_resources.files("smif").joinpath("sample_project")
    _copy_traversable(root, directory)
    if directory == ".":
        dirname = "the current directory"
    else:
        dirname = directory
    logging.info("Created sample project in %s", dirname)


def _copy_traversable(traversable, dst):
    """Recursively copy package resources to filesystem."""
    if traversable.is_dir():
        os.makedirs(dst, exist_ok=True)
        for child in traversable.iterdir():
            _copy_traversable(child, os.path.join(dst, child.name))
    else:
        # Ensure parent directory exists
        parent = os.path.dirname(dst)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        # Write bytes from the resource to the destination file
        data = traversable.read_bytes()
        with open(dst, "wb") as fh:
            fh.write(data)
