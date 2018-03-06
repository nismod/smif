import React, { Component } from 'react'
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

                for (var key in sosModelRun.scenarios) {

                    if ((scenarioSet == key) && (selectedScenarios[scenarioSet][i].name == sosModelRun.scenarios[key])) {
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

        if (this.props.sosModelRun.scenarios[target.name] == target.value) {
            onChange(target.name, "")
        } else {
            onChange(target.name, target.value)
        }
    }

    renderScenarioSelector(selectedScenarios) {

        return (
            <div>
                {
                    Object.keys(selectedScenarios).map((scenarioSet) => (
                        <div key={scenarioSet}>
                            <div className="card">
                                <div className="card-body">
                                    <label className="card-title">{scenarioSet}</label>
                                    {
                                        selectedScenarios[scenarioSet].map((scenario) => (
                                            <div className="form-check" key={scenario.name}>
                                                <label className="form-check-label">
                                                    <input id={'radio_' + scenarioSet + '_' + scenario.name} className="form-check-input" type="radio" name={scenarioSet} key={scenario.name} value={scenario.name} checked={scenario.active} onClick={this.handleChange}/>
                                                    {scenario.name}
                                                </label>
                                            </div>
                                        ))
                                    }
                                </div>
                            </div>
                            <br/>
                        </div>
                    ))
                }
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="scenario_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="scenario_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="scenario_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {sosModelRun, sosModels, scenarios} = this.props

        let selectedScenarios = null
        let selectedSosModel = null

        if (sosModelRun == null || sosModelRun == undefined || Object.keys(sosModelRun).length == 0) {
            return this.renderDanger('There is no SosModelRun configured')
        } else if (sosModels == null || sosModels == undefined || sosModels[0] == null) {
            return this.renderDanger('There are no SosModels configured')
        } else if (scenarios == null || scenarios == undefined || scenarios[0] == null) {
            return this.renderDanger('There are no Scenarios configured')
        } else if (sosModelRun.sos_model == "" || sosModelRun.sos_model == null || sosModelRun.sos_model == undefined) {
            return this.renderInfo('There is no SosModel selected in the SosModelRun')
        } else {
            selectedSosModel = this.pickSosModelByName(sosModelRun.sos_model, sosModels)
            if (selectedSosModel.scenario_sets == null || selectedSosModel.scenario_sets == undefined || selectedSosModel.scenario_sets[0] == undefined) {
                return this.renderInfo('There are no ScenarioSets configured in the SosModel')
            }

            selectedScenarios = this.pickScenariosBySets(selectedSosModel.scenario_sets, scenarios)
            selectedScenarios = this.flagActiveScenarios(selectedScenarios, sosModelRun)

            return this.renderScenarioSelector(selectedScenarios)
        }
    }
}

ScenarioSelector.propTypes = {
    sosModelRun: PropTypes.object,
    sosModels: PropTypes.array,
    scenarios: PropTypes.array,
    onChange: PropTypes.func
}

export default ScenarioSelector
