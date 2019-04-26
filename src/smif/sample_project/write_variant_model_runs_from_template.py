"""
Script that generate nb_model_runs model run config files from a template 
model run file template_model_run, for each nb_model_runs variants of a scenario.

Command line arguments:
----------------------
template_model_run: name of the template file
scenario_name: Name of the scenario that is varied
nb_model_runs: number of generated model run files



"""

import sys
import warnings
from smif.data_layer.file import YamlConfigStore

nb_model_runs_default = 3
nb_model_runs = nb_model_runs_default

nb_cl_args = len(sys.argv)
if nb_cl_args>1:
    # cl args must contain model run AND scenario, nb_model_runs optional
    assert(nb_cl_args>2), "Cannot provide only one argument to script"
    template_model_run, scenario_name = (str(sys.argv[1]), str(sys.argv[2]))
else:
    template_model_run = 'energy_central'
    scenario_name = 'water_sector_energy_demand'

if nb_cl_args<4:
    warnings.warn("Unknown number of replicates: default to {}".format(nb_model_runs_default))
    
    
my_config_store = YamlConfigStore('./')

model_run = my_config_store.read_model_run(template_model_run)
assert(scenario_name in model_run['scenarios']), "Error: Unknown scenario"
   
template_model_run = model_run['name']

# Open batchfile
f_handle = open(template_model_run+'.batch', 'w')
""" For each variant model_run, write a new model run file with corresponding
    scenario variant and update batchfile.
"""
for i in range(0,nb_model_runs):
    model_run_name = template_model_run+'_{:d}'.format(i)
    model_run['name'] = model_run_name
    model_run['scenarios'][scenario_name] = 'replicate_{:d}'.format(i)
    my_config_store.write_model_run(model_run)
    f_handle.write(model_run_name+'\n')
    
f_handle.close()
