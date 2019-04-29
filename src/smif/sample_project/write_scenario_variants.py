"""
Generate multiple scenario variants given an initial .yml scenario template.

Command line arguments
-----------------------
scenario_file: .yml file serving as a template
nb_variants: nb of scenario variants
"""

import os

from smif.data_layer.file import YamlConfigStore

scenario_file = sys.argv[1]
nb_variants = sys.argv[2]

my_config_store = YamlConfigStore('./')

scenario_name, ext = os.path.splitext(scenario_file)
full_path = os.path.join(my_config_store.config_folders['scenarios'], scenario_name)
copy_template_file = 'cp '+full_path+'.yml '+full_path+'_template.yml'

os.system(copy_template_file)

scenario = my_config_store.read_scenario(scenario_name)
root = {}
variant = my_config_store.read_scenario_variant(scenario_name,
                                                         scenario_name+'_variant')
for output in scenario['provides']:
    root[output['name']], ext = os.path.splitext(variant['data'][output['name']])
        

first_variant = True
for ivar in range(0,nb_variants):
    for output in scenario['provides']:
        variant['name'] = scenario_name+'_variant_{:d}'.format(ivar)
        variant['data'][output['name']] = root[output['name']]+'{:d}'.format(ivar)+ext
        
    if(first_variant):
        first_variant = False
        my_config_store.update_scenario_variant(scenario_name, scenario_name+'_variant', variant)
    else:
        my_config_store.write_scenario_variant(scenario_name, variant)



        
    
