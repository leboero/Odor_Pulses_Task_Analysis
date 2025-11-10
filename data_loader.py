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

# ---------------------------
# CONFIG CLASS
# ---------------------------
class LoaderConfig:
    def __init__(
        self,
        root_dir: str = "Session", #folder in which the 'session_..._.pkl' files with the data from the behavioral session are.
        animals: Optional[List[str]] = None, #List of strings of the animal names you want to load
        date_range: Optional[Tuple[int, int]] = None,
        filters: Optional[Dict[str, Any]] = None,
        load_sniff: bool = True, #As of now, whether you want to get the histogram showing at which phase bin pulses arrived in each trial (sniff_histogram)
        load_breathing: bool = True, #Whether you want to load the breathing trace or not
        #load_autocorr: bool = True, #For incoproating later.
        verbose: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.animals = animals  # e.g., ['Tabby', 'Banner']
        self.date_range = date_range  # (20190419, 20220526)
        self.filters = filters or {}  # ratio of lambdas, sampling window duration, performance, etc.
        self.load_sniff = load_sniff
        self.load_breathing = load_breathing
        #self.load_autocorr = load_autocorr #not for now, but can be added later.
        self.verbose = verbose

def scan_session_metadata(file_path: Path) -> Dict[str, Any]:
    """Open a file just enough to read identifying and lightweight variables."""
    try:
        with open(file_path, "rb") as f:
            session_list = pickle.load(f)
            session = session_list[0]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
    
    year = session.get("year")
    month = session.get("month")
    day = session.get("day")
    animal_ = file_path.name[8:-18]

    # Deal with files without full date info
    if None in (year, month, day):
        year = int(file_path.name[-17:-13])
        month = str(file_path.name[-13:-11])
        day = str(file_path.name[-11:-9])
        meta = {
        "animal": str(animal_),  #session.get("animal"), we could also use this, but I prefer to use directly the filename because 
        # if the file was saved accidentally with a wrong animal ID, the session file won't have the corrected information even if the filename does
        "date": int(str(str(year)+month+day)),
        "session_num": session.get("session_id"),
        "num_trials": session.get("num_trials", 0),
        "type": session.get("type", None), #to select session with 'random' or 'block' trials
        "delay_time": session.get("delay_time", None), #duration of the sampling window
        "high_to_low_ratio": (round(session["high_count"]/session["low_count"])), #ratio between the lambda for the 'high' and 'low' trials. 
    }
        
    else:
        meta = {
        "animal": str(animal_),  #session.get("animal"), we could also use this, but I prefer to use directly the filename because 
        # if the file was saved accidentally with a wrong animal ID, the session file won't have the corrected information even if the filename does
        "date": int(f"{session.get('year'):04d}{session.get('month'):02d}{session.get('day'):02d}"),
        "session_num": session.get("session_id"),
        "num_trials": session.get("num_trials", 0),
        "type": session.get("type", None), #to select session with 'random' or 'block' trials
        "delay_time": session.get("delay_time", None), #duration of the sampling window
        "high_to_low_ratio": (round(session["high_count"]/session["low_count"])), #ratio between the lambda for the 'high' and 'low' trials. 
    }
    return meta

def passes_filters(meta: Dict[str, Any], config: LoaderConfig) -> bool: #Function for passing the filters to select which sessions to load.
    """Check both file-level and metadata-level filters."""
    if not meta:
        return False

    # Filter by animal
    if config.animals and meta["animal"] not in config.animals:
        return False

    # Filter by date range
    if config.date_range:
        start, end = config.date_range
        if not (start <= meta["date"] <= end):
            return False

    ntrials_min = config.filters.get("n_trials_min")
    if ntrials_min and meta["num_trials"] < ntrials_min:
        return False

    # Example of experiment-type filter
    if config.filters.get("type") and meta["type"] != config.filters["type"]:
        return False

    if config.filters.get("delay_time") and meta["delay_time"] != config.filters["delay_time"]:
        return False

    if config.filters.get("high_to_low_ratio") and meta["high_to_low_ratio"] != config.filters["high_to_low_ratio"]:
        return False

    return True

# ---------------------------
# FILE PARSING HELPERS
# ---------------------------
def parse_session_filename(filename: str):
    """
    Parse filenames like:
    session_MouseName_20231011_0.pickle
    Returns: (animal, date, session_num)
    """
    pattern = r"session_([A-Za-z0-9]+)_(\d{8})_(\d+)\.pickle"
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    animal, date, session_num = m.groups()
    return animal, date, int(session_num)

# ---------------------------
# SESSION LOADER
# ---------------------------
def load_session_file(file_path: Path, config: LoaderConfig) -> Optional[Dict[str, Any]]:
    """Fully load one session file and compute all desired variables."""
    try:
        with open(file_path, "rb") as f:
            session_list = pickle.load(f)
            session = session_list[0]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Define non-idle trials
    non_idle_trials = np.invert(session["idle_trials"])
    #non_early_trials = np.invert(session['early_trials'])
    valid_trials = non_idle_trials #& non_early_trials
    high_trials = session['high_trials'][valid_trials]
    trial_odor = session['trial_odor'][valid_trials]
    cum_odor = np.ceil(trial_odor.sum(axis=1)/(session['pulse_time_ms']*100)) #session['level1'] can be used for newer versions

    # This is only for Bengal, SHorthair, Tabby (and maybe Banner) since the code for exporting session files did not account for early trials, 
    # and early trials would be considered non_idle but w random assignment  of high/low

    high_seven = cum_odor>8
    low_eight = cum_odor<=8
    all_low_trials = np.invert(high_trials.astype(np.bool))
    bad_trials_1 = np.logical_and(high_seven,all_low_trials)
    bad_trials_2 = np.logical_and(low_eight,high_trials)
    bad_trials = np.logical_or(bad_trials_1,bad_trials_2)
    good_trials = np.invert(bad_trials) #This would be a convoluted way of filtering 'early' trials.
    
    data = {
        "date": str(file_path.name[-17:-9]),
        "num_trials": session["num_trials"],
        "sampling_window": session['delay_time']*1000, #in ms
        "type": session['type'], #random or block
        "ratio": np.round(session['high_count']/session['low_count']),
        "good_trials": good_trials,
        "correct_trials": session["correct_trials"][valid_trials][good_trials],
        "high_trials": session["high_trials"][valid_trials][good_trials],
        "low_trials": session["low_trials"][valid_trials][good_trials],
        "high_choices": (session["correct_trials"][valid_trials][good_trials]==session["high_trials"][valid_trials][good_trials]),
        "trial_odor": trial_odor[good_trials],
        "cum_odor": cum_odor[good_trials]
    }
    if config.load_sniff:
        #shh = get_sniff_histogram(session, False)

        data["sniff_hist"] = get_sniff_histogram(session, False)[valid_trials][good_trials]
        data["sniff_hist_shuffled"] = get_sniff_histogram(session, True)[valid_trials][good_trials]

    if config.load_breathing:
        data["breathing"] = np.append(session['trial_pre_breath'][valid_trials, -2500:][good_trials], session['trial_breath'][valid_trials, :6500][good_trials], axis=1)

    # if config.load_spikes:
    #     data["spikes"] = load_spike_data(session_data)

    return data

# ---------------------------
# MAIN LOADER
# ---------------------------
def run_loader(config: LoaderConfig) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all session files passing the pre-filters."""
    all_data = {}

    for file_path in config.root_dir.glob("session_*.pickle"):
        meta = scan_session_metadata(file_path)

        if not passes_filters(meta, config):
            continue

        animal = meta["animal"]
        date = str(meta["date"])
        session_num = str(meta["session_num"])

        if config.verbose:
            print(f"Loading: {animal} | {date} | session {session_num}")

        session_data = load_session_file(file_path, config)
        if session_data is None:
            continue

        all_data.setdefault(animal, {}).setdefault(date, {})[session_num] = session_data

    return all_data

def save_hdf5(filename, data):
    with h5py.File(filename, "w") as f:
        def recurse(h5group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    subgroup = h5group.create_group(k)
                    recurse(subgroup, v)
                else:
                    try:
                        h5group.create_dataset(k, data=np.array(v))
                    except Exception:
                        # store non-numeric data as string
                        h5group.create_dataset(k, data=str(v))
        recurse(f, data)