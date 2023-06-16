# -*- coding: utf-8 -*-


import os
import msm_lib as msm


config_path = os.path.join('serial-arena', 'config.json')
# config_path = os.path.join('parallel-arena', 'config.json')
# config_path = os.path.join('cyclic-arena', 'config.json')
config = msm.load_config(config_path)
log = msm.load_log(config)
model = msm.generate_model(log, config)
msm.save_model(model, config)
model = msm.load_model(config)
msm.show_model(model)
