import os
import pickle
from pathlib import Path
import pickle
import re
import pandas as pd
import numpy as np
import h5py
import tqdm
import scipy
from scipy import signal
from tqdm import tnrange
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
import sys
import warnings
from odor_breathing_functions import*
warnings.filterwarnings('ignore')

def load_hdf5(filename):
    def recurse(h5group):
        d = {}
        for key, item in h5group.items():
            if isinstance(item, h5py.Group):
                d[key] = recurse(item)
            else:
                data = item[()]
                # Decode bytes or convert arrays
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                elif isinstance(data, np.ndarray) and data.shape == ():
                    data = data.item()  # convert 0-D arrays to scalars
                d[key] = data
        return d

    with h5py.File(filename, "r") as f:
        return recurse(f)
    
def pool_variables(all_data, var_names, by_animal=True):
    """
    Pools specified variables from a nested dictionary structure:
    all_data[animal][date][session_num][var_name]

    Parameters
    ----------
    all_data : dict
        Nested dictionary of the form {animal: {date: {session_num: data_dict}}}
    var_names : list of str
        Variables to pool across sessions.
    by_animal : bool, optional
        If True, returns data pooled separately per animal.
        If False, returns all data pooled together.

    Returns
    -------
    pooled : dict
        Dictionary with pooled arrays per variable (and per animal if by_animal=True).
    """
    pooled = {v: {} if by_animal else [] for v in var_names}

    for var in var_names:
        for animal, date_dict in all_data.items():
            vals = []

            for date, sessions in date_dict.items():
                for session_num, data in sessions.items():
                    if var in data and data[var] is not None:
                        vals.append(np.asarray(data[var]))

            if not vals:
                continue

            if by_animal:
                pooled[var][animal] = np.concatenate(vals)
            else:
                # If pooling across all animals
                pooled[var].extend(vals)

        # After all animals, concatenate if not by_animal
        if not by_animal and pooled[var]:
            pooled[var] = np.concatenate(pooled[var])

    return pooled

def convert_pooled_to_array(all_data, var_name, by_animal=True, fill_value=np.nan):
    """
    Convert nested all_data dict to a numpy array pooling a specific variable across sessions.

    Structure expected:
    all_data[animal][date][session_num][var_name] = array-like

    Parameters
    ----------
    all_data : dict
        Nested dictionary of structure {animal: {date: {session_num: data_dict}}}
    var_name : str
        Variable name to extract and pool (e.g., 'sniff_hist')
    by_animal : bool, optional
        If True, returns a 3D array (animal x trials x features)
        If False, returns a 2D array (trials x features)
    fill_value : float, optional
        Value used to pad when trial counts differ between animals (default: np.nan)

    Returns
    -------
    pooled : np.ndarray
        2D (if by_animal=False) or 3D (if by_animal=True) numpy array with pooled data.
    animal_order : list
        List of animal keys corresponding to the first axis (if by_animal=True).
    """
    # ---- Helper to extract one variable for one animal ----
    def extract_animal_data(animal_dict, var_name):
        vals = []
        for date, sessions in animal_dict.items():
            for session_num, data in sessions.items():
                if var_name in data and data[var_name] is not None:
                    vals.append(np.asarray(data[var_name]))
        return np.concatenate(vals, axis=0) if vals else np.empty((0,))

    animals = list(all_data.keys())

    if by_animal:
        pooled_per_animal = []
        max_trials = 0
        feature_dim = None

        # Collect data per animal
        for animal in animals:
            arr = extract_animal_data(all_data[animal], var_name)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            pooled_per_animal.append(arr)
            max_trials = max(max_trials, arr.shape[0])
            feature_dim = arr.shape[1]

        # Pad to make equal trial counts
        pooled = np.full((len(animals), max_trials, feature_dim), fill_value)
        for i, arr in enumerate(pooled_per_animal):
            pooled[i, :arr.shape[0], :] = arr

        return pooled, animals

    else:
        # Pool across all animals
        all_vals = []
        for animal in animals:
            arr = extract_animal_data(all_data[animal], var_name)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            all_vals.append(arr)

        if all_vals:
            pooled = np.concatenate(all_vals, axis=0)
        else:
            pooled = np.empty((0,))

        return pooled, None
