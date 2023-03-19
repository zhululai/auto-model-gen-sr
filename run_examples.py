# -*- coding: utf-8 -*-


import os
import msm_lib as msm


config_path = os.path.join('serial-arena', 'config.json')
# config_path = os.path.join('parallel-arena', 'config.json')
# config_path = os.path.join('cyclic-arena', 'config.json')
config = msm.load_config(config_path)
log = msm.load_log(config)
graph = msm.generate_graph(log, config)
msm.save_graph(graph, config)
graph = msm.load_graph(config)
msm.show_graph(graph)
