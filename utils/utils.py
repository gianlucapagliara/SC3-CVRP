import pandas as pd
import yaml
from easydict import EasyDict


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    """

    # parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.safe_load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print("INVALID YAML file format! Please provide a good yaml file.")
            exit()


def load_data(file_path):
    """
    Load the data from file_path and preprocess them.
    """
    df = pd.read_csv(file_path)
    df['detected_at'] = pd.to_datetime(df['detected_at'], format='%d/%m/%Y %H:%M')
    df.occluded.replace({1: False, 2: True}, inplace=True)
    df.fillna(value=False, inplace=True)
    df = df[df.anomaly_type == False]
    df.drop('anomaly_type', axis=1, inplace=True)
    df.set_index('detected_at', inplace=True, drop=True)
    df.sort_index(inplace=True)

    return df


def filter_by_level(data, level=3):
    """
    Filter the bins to empty at every window using the filling level.
    """
    new_data = data.drop_duplicates(subset='bin_serial', keep='last')
    new_data = new_data[(new_data.bin_level > level) | new_data.occluded]
    return new_data
