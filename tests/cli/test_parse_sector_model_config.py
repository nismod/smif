# -*- coding: utf-8 -*-
import os
from smif.cli.parse_sector_model_config import SectorModelReader

class TestSectorModelReader(object):
    def _model_config_dir(self, project_folder):
        return os.path.join(str(project_folder), 'data', 'water_supply')


