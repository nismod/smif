import React, { Component } from 'react';
import PropTypes from 'prop-types'

class ScenarioSelector extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
    }

    pickSosModelByName(sos_model_name, sos_models) {
        /**
         * Get SosModel parameters, that belong to a given sos_model_name
         * 
         * Arguments
         * ---------
         * sos_model_name: str
         *     Name identifier of the sos_model
         * sos_models: array
         *     Full list of available sos_models
         * 
         * Returns
         * -------
         * Object
         *     All sos_model parameters that belong to the given sos_model_name
         */

        let sos_model = sos_models.filter(
            (sos_model) => sos_model.name === sos_model_name
        )[0]

        if (typeof sos_model === 'undefined') {
            sos_model = sos_models[0]
        }
        
        return sos_model
    }

    pickScenariosBySets(scenario_sets, scenarios) {
        /** 
         * Get all the scenarios, that belong to a given scenario_sets
         * 
         * Arguments
         * ---------
         * scenario_sets: str
         *     Name identifier of the scenario_sets
         * scenarios: array
         *     Full list of available scenarios
         * 
         * Returns
         * -------
         * Object
         *     All scenarios that belong to the given scenario_sets
         */ 

        let scenarios_in_sets = new Object()

        for (let i = 0; i < scenario_sets.length; i++) {

            // Get all scenarios that belong to this scenario set
            scenarios_in_sets[scenario_sets[i]] = scenarios.filter(scenarios => scenarios.scenario_set === scenario_sets[i])

        }
        return scenarios_in_sets
    }

    flagActiveScenarios(selectedScenarios, sosModelRun) {
        /**
         * Flag the scenarios that are active in the project configuration
         * 
         * Arguments
         * ---------
         * 
         * Returns
         * -------
         * Object
         *     All scenarios complimented with a true or false active flag
         */

        Object.keys(selectedScenarios).forEach(function(scenarioSet) {
            for (let i = 0; i < selectedScenarios[scenarioSet].length; i++) {
                selectedScenarios[scenarioSet][i].active = false

                for (let k = 0; k < sosModelRun.scenarios.length; k++) {

                    let obj = {
                        [scenarioSet]: selectedScenarios[scenarioSet][i].name
                    }
                    if (JSON.stringify(obj) === JSON.stringify(sosModelRun.scenarios[k])) {
                        selectedScenarios[scenarioSet][i].active = true
                    }
                }
            }
        })

        return selectedScenarios
    }

    handleChange(event) {
        const target = event.target
        const {onChange} = this.props

        onChange(target.name, target.value)
    }

    render() {
        const {sosModelRun, sosModels, scenarios} = this.props

        let selectedSosModel = null
        let selectedScenarios = null

        if ((sosModelRun && sosModelRun.scenarios) && (sosModels.length > 0) && (scenarios.length > 0)) {

            selectedSosModel = this.pickSosModelByName(sosModelRun.sos_model, sosModels)
            selectedScenarios = this.pickScenariosBySets(selectedSosModel.scenario_sets, scenarios)
            selectedScenarios = this.flagActiveScenarios(selectedScenarios, sosModelRun)                
        }

        return (

            <div>
                {
                    Object.keys(selectedScenarios).map((scenarioSet) => (
                        <fieldset key={scenarioSet}>
                            <legend>{scenarioSet}</legend>
                            {
                                selectedScenarios[scenarioSet].map((scenario) => (
                                    <div key={scenario.name}>
                                        <input type="radio" name={scenarioSet} key={scenario.name} value={scenario.name} defaultChecked={scenario.active} onClick={this.handleChange}></input>
                                        <label>{scenario.name}</label>
                                    </div>
                                ))
                            }
                        </fieldset>
                    ))
                }
            </div>
        )
    }
}

ScenarioSelector.propTypes = {
    sosModelRun: PropTypes.object,
    sosModels: PropTypes.array,
    scenarios: PropTypes.array,
    onChange: PropTypes.func
};

export default ScenarioSelector;