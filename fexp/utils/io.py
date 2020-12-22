# coding=utf-8
"""
Copyright (c) Fexp Contributors

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import json
import pathlib


def write_list(filename, data):
    """
    Write list line-by-line to file.

    Parameters
    ----------
    filename : str or pathlib.Path
    data : list
    """
    with open(filename, "w") as f:
        for line in data:
            f.write(str(line) + "\n")


def read_list(filename):
    """
    Read file line by line, ignoring lines starting with '#'

    Parameters
    ----------
    filename : str or pathlib.Path

    Returns
    -------
    list
    """

    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines if not line.startswith("#")]


def write_json(filename, data, indent=2):
    """
    Write data to a json file

    Parameters
    ----------
    filename : str or pathlib.Path
    data : dict
        JSON serializable dict
    indent : int
        Indent to make file human readable.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(filename):
    """
    Read json from file

    Parameters
    ----------
    filename : str of pathlib.Path

    Returns
    -------
    dict
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data
