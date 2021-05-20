"""
A File that should contain only the (single) global_env_prop variables
This variable is shared by all files in the project.
Don't change this file!
"""
import os

import Utilities.Configuration.config as cfg
from arguments import SchemaName
from gym_atena.lib.flight_delays_helpers import create_flights_env_properties
from gym_atena.lib.networking_helpers import create_networking_env_properties
from gym_atena.lib.big_flights_helpers import create_big_flights_env_properties
from gym_atena.lib.wide_flights_helpers import create_wide_flights_env_properties
from gym_atena.lib.wide12_flights_helpers import create_wide12_flights_env_properties
from gym_atena.lib.netflix_helpers import create_netflix_env_properties

global_env_prop = None


def update_global_env_prop_from_cfg():
    """
    Changing the properties of the current schema based on cfg.schema and returning the new properties object
    Returns:

    """
    global global_env_prop
    schema_name = SchemaName(cfg.schema)
    if schema_name is SchemaName.NETWORKING:
        global_env_prop = create_networking_env_properties()
    elif schema_name is SchemaName.FLIGHTS:
        global_env_prop = create_flights_env_properties()
    elif schema_name is SchemaName.BIG_FLIGHTS:
        global_env_prop = create_big_flights_env_properties()
    elif schema_name is SchemaName.WIDE_FLIGHTS:
        global_env_prop = create_wide_flights_env_properties()
    elif schema_name is SchemaName.WIDE12_FLIGHTS:
        global_env_prop = create_wide12_flights_env_properties()
    elif schema_name is SchemaName.NETFLIX:
        global_env_prop = create_netflix_env_properties()
    else:
        raise NotImplementedError
    if cfg.dataset_number is not None and cfg.outdir:
        # Save dataset name
        with open(os.path.join(cfg.outdir, 'dataset.txt'), 'w') as f:
            f.write(str(global_env_prop.env_dataset_prop.repo.file_list[cfg.dataset_number]))
    return global_env_prop


update_global_env_prop_from_cfg()
