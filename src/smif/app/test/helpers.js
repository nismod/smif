// setup used by all tests
import { configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-15';

configure({ adapter: new Adapter() });

// data used by all tests
export const empty_object = {}
export const empty_array = []

export const sos_model_runs = [
    {
        description: 'Combined energy and water under central scenario, only to 2015',
        name: '20170918_energy_water_short'
    },
    {
        description: 'Combined energy and water under central scenario',
        name: '20170918_energy_water'
    }
]

export const sos_model_run = {
    decision_module: '',
    description: 'Combined energy and water under central scenario, only to 2015',
    name: '20170918_energy_water_short',
    narratives: {
        technology: [
            'High Tech Demand Side Management'
        ]
    },
    scenarios: {
        population: 'Central Population (Medium)',
        rainfall: 'Central Rainfall'
    },
    sos_model: 'energy_waste',
    stamp: 'Mon, 18 Sep 2017 12:53:23 GMT',
    timesteps: [
        2010,
        2015
    ]
}

export const sos_models = [
    {
        convergence_absolute_tolerance: '1e-05',
        convergence_relative_tolerance: '1e-05',
        dependencies: [
            {
                sink_model: 'water_supply',
                sink_model_input: 'raininess',
                source_model: 'rainfall',
                source_model_output: 'raininess'
            },
            {
                sink_model: 'water_supply',
                sink_model_input: 'population',
                source_model: 'population',
                source_model_output: 'population'
            },
            {
                sink_model: 'energy_demand',
                sink_model_input: 'energy_demand',
                source_model: 'water_supply',
                source_model_output: 'energy_demand'
            },
            {
                sink_model: 'energy_demand',
                sink_model_input: 'population',
                source_model: 'population',
                source_model_output: 'population'
            },
            {
                sink_model: 'water_supply',
                sink_model_input: 'water_demand',
                source_model: 'energy_demand',
                source_model_output: 'water_demand'
            }
        ],
        description: 'The future supply and demand of energy and water for the UK',
        max_iterations: 100,
        name: 'energy_waste',
        narrative_sets: [
            'technology'
        ],
        scenario_sets: [
            'population',
            'rainfall'
        ],
        sector_models: [
            'water_supply',
            'energy_demand'
        ]
    },
    {
        convergence_absolute_tolerance: '1e-05',
        convergence_relative_tolerance: '1e-05',
        dependencies: [
            {
                sink_model: 'water_supply',
                sink_model_input: 'raininess',
                source_model: 'rainfall',
                source_model_output: 'raininess'
            },
            {
                sink_model: 'water_supply',
                sink_model_input: 'population',
                source_model: 'population',
                source_model_output: 'population'
            },
            {
                sink_model: 'energy_demand',
                sink_model_input: 'energy_demand',
                source_model: 'water_supply',
                source_model_output: 'energy_demand'
            },
            {
                sink_model: 'energy_demand',
                sink_model_input: 'population',
                source_model: 'population',
                source_model_output: 'population'
            },
            {
                sink_model: 'water_supply',
                sink_model_input: 'water_demand',
                source_model: 'energy_demand',
                source_model_output: 'water_demand'
            }
        ],
        description: 'The future supply and demand of energy and water for the UK',
        max_iterations: 100,
        name: 'energy_water',
        scenario_sets: [
            'population'
        ],
        sector_models: [
            'water_supply',
            'energy_demand'
        ]
    }
]

export const sos_model = {
    convergence_absolute_tolerance: '1e-05',
    convergence_relative_tolerance: '1e-05',
    dependencies: [
        {
            sink_model: 'water_supply',
            sink_model_input: 'raininess',
            source_model: 'rainfall',
            source_model_output: 'raininess'
        },
        {
            sink_model: 'water_supply',
            sink_model_input: 'population',
            source_model: 'population',
            source_model_output: 'population'
        },
        {
            sink_model: 'energy_demand',
            sink_model_input: 'energy_demand',
            source_model: 'water_supply',
            source_model_output: 'energy_demand'
        },
        {
            sink_model: 'energy_demand',
            sink_model_input: 'population',
            source_model: 'population',
            source_model_output: 'population'
        },
        {
            sink_model: 'water_supply',
            sink_model_input: 'water_demand',
            source_model: 'energy_demand',
            source_model_output: 'water_demand'
        }
    ],
    description: 'The future supply and demand of energy and water for the UK',
    max_iterations: 100,
    name: 'energy_waste',
    narrative_sets: [
        'technology'
    ],
    scenario_sets: [
        'population',
        'rainfall'
    ],
    sector_models: [
        'water_supply',
        'energy_demand'
    ]
}

export const sector_models = [
    {
        classname: 'WaterSupplySectorModel',
        description: 'Simulates the optimal operation of the UK water supply system',
        initial_conditions: [
            'water_supply_oxford.yml',
            'reservoirs.yml'
        ],
        inputs: [
            {
                name: 'raininess',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'ml'
            },
            {
                name: 'population',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million people'
            },
            {
                name: 'water_demand',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'Ml'
            }
        ],
        interventions: [
            'water_supply.yml'
        ],
        name: 'water_supply',
        outputs: [
            {
                name: 'cost',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million GBP'
            },
            {
                name: 'energy_demand',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'kWh'
            },
            {
                name: 'water',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'Ml'
            }
        ],
        parameters: [
            {
                absolute_range: '(0, 100)',
                default_value: 3,
                description: 'The savings from smart water meters',
                name: 'clever_water_meter_savings',
                suggested_range: '(3, 10)',
                units: '%'
            }
        ],
        path: 'models/water_supply.py'
    },
    {
        classname: 'EDMWrapper',
        description: '',
        initial_conditions: [],
        inputs: [
            {
                name: 'population',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million people'
            },
            {
                name: 'energy_demand',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'kWh'
            }
        ],
        interventions: [],
        name: 'energy_demand',
        outputs: [
            {
                name: 'cost',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million GBP'
            },
            {
                name: 'water_demand',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'Ml'
            }
        ],
        parameters: [
            {
                absolute_range: '(0, 100)',
                default_value: 3,
                description: 'The savings from smart meters',
                name: 'smart_meter_savings',
                suggested_range: '(3, 10)',
                units: '%'
            }
        ],
        path: 'models/energy_demand.py'
    }
]

export const sector_model = {
    classname: 'EDMWrapper',
    description: '',
    initial_conditions: [],
    inputs: [
        {
            name: 'population',
            spatial_resolution: 'national',
            temporal_resolution: 'annual',
            units: 'million people'
        },
        {
            name: 'energy_demand',
            spatial_resolution: 'national',
            temporal_resolution: 'annual',
            units: 'kWh'
        }
    ],
    interventions: [],
    name: 'energy_demand',
    outputs: [
        {
            name: 'cost',
            spatial_resolution: 'national',
            temporal_resolution: 'annual',
            units: 'million GBP'
        },
        {
            name: 'water_demand',
            spatial_resolution: 'national',
            temporal_resolution: 'annual',
            units: 'Ml'
        }
    ],
    parameters: [
        {
            absolute_range: '(0, 100)',
            default_value: 3,
            description: 'The savings from smart meters',
            name: 'smart_meter_savings',
            suggested_range: '(3, 10)',
            units: '%'
        }
    ],
    path: 'models/energy_demand.py'
}


export const scenario_sets = [
    {
        description: 'UK precipitation',
        name: 'rainfall',
        facets: [
            {
                description: 'Rainfall for the UK',
                name: 'rainfall'
            }
        ]

    },
    {
        description: 'Growth in UK population',
        name: 'population',
        facets: [
            {
                description: 'Central Population for the UK',
                name: 'population'
            }
        ]
    }
]

export const scenario_set = {
    description: 'Growth in UK population',
    name: 'population',
    facets: [
        {
            description: 'Central Population for the UK',
            name: 'population'
        }
    ]
}

export const scenarios = [
    {
        description: 'Central Population for the UK (Low)',
        name: 'Central Population (Low)',
        facets: [
            {
                filename: 'population_low.csv',
                name: 'population',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million people'
            }
        ],
        scenario_set: 'population',
        active: false
    },
    {
        description: 'Central Population for the UK (Medium)',
        name: 'Central Population (Medium)',
        facets: [
            {
                filename: 'population_med.csv',
                name: 'population',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million people'
            }
        ],
        scenario_set: 'population',
        active: false
    },
    {
        description: 'Central Population for the UK (High)',
        name: 'Central Population (High)',
        facets: [
            {
                filename: 'population_low.csv',
                name: 'population',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'million people'
            }
        ],
        scenario_set: 'population',
        active: false
    },
    {
        description: 'Central Rainfall Scenario for the UK',
        name: 'Central Rainfall',
        facets: [
            {
                filename: 'raininess.csv',
                name: 'raininess',
                spatial_resolution: 'national',
                temporal_resolution: 'annual',
                units: 'ml'
            }
        ],
        scenario_set: 'rainfall'
    }
]

export const scenario = {
    description: 'Central Population for the UK (Medium)',
    name: 'Central Population (Medium)',
    facets: [
        {
            filename: 'population_med.csv',
            name: 'population',
            spatial_resolution: 'national',
            temporal_resolution: 'annual',
            units: 'million people'
        }
    ],
    scenario_set: 'population'
}

export const narrative_sets = [
    {
        description: 'Describes the evolution of technology',
        name: 'technology'
    }
]

export const narrative_set = {
    description: 'Describes the evolution of technology',
    name: 'technology'
}

export const narratives = [
    {
        description: 'High penetration of SMART technology on the demand side',
        filename: 'high_tech_dsm.yml',
        name: 'High Tech Demand Side Management',
        narrative_set: 'technology'
    },
    {
        description: 'Low penetration of SMART technology on the demand side',
        filename: 'low_tech_dsm.yml',
        name: 'Low Tech Demand Side Management',
        narrative_set: 'technology'
    }
]

export const narrative = {
    description: 'High penetration of SMART technology on the demand side',
    filename: 'high_tech_dsm.yml',
    name: 'High Tech Demand Side Management',
    narrative_set: 'technology'
}
