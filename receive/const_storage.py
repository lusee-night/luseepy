#!python3

from typing import Optional
import yaml

class Constants:
    def __init__(self, config_path: Optional[str]=None):
        if config_path is None:
            self.lun_lat_deg = -15.0
            self.lun_long_deg = 175.0
            self.lun_height_m = 0.0
        else:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            self.lun_lat_deg = float(data["lun_lat_deg"])
            self.lun_long_deg = float(data["lun_long_deg"])
            self.lun_height_m = float(data["lun_height_m"])
