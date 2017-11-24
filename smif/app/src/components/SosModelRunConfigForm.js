import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Link } from 'react-router-dom'
import update from 'react-addons-update';

import { saveSosModelRun } from '../actions/actions.js';

import ScenarioSelector from '../components/ScenarioSelector.js'
import NarrativeSelector from '../components/NarrativeSelector.js'
import TimestepSelector from '../components/TimestepSelector.js'

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props)

        this.selectSosModel = this.selectSosModel.bind(this)
        this.pickSosModelByName = this.pickSosModelByName.bind(this)
        this.handleScenariosChange = this.handleScenariosChange.bind(this)
        this.handleNarrativeChange = this.handleNarrativeChange.bind(this)
        this.handleTimestepChange = this.handleTimestepChange.bind(this)
        this.handleInputChange = this.handleInputChange.bind(this)
        this.handleSave = this.handleSave.bind(this)

        this.state = {}
        this.state.selectedSosModelRun = this.props.sos_model_run
        this.state.selectedSosModel = this.pickSosModelByName(this.props.sos_model_run.sos_model)

        this.state.selectedScenarios = this.pickScenariosBySet(this.state.selectedSosModel.scenario_sets)
        this.state.selectedNarratives = this.pickNarrativesBySet(this.state.selectedSosModel.narrative_sets)
    }

    pickSosModelByName(sos_model_name) {
        /**
         * Get SosModel parameters, that belong to a given sos_model_name
         * 
         * Arguments
         * ---------
         * sos_model_name: str
         *     Name identifier of the sos_model
         * 
         * Returns
         * -------
         * Object
         *     All sos_model parameters that belong to the given sos_model_name
         */

        let sos_model = this.props.sos_models.filter(
            (sos_model) => sos_model.name === sos_model_name
        )[0]

        if (typeof sos_model === 'undefined') {
            sos_model = this.props.sos_models[0]
        }
        
        return sos_model
    }
    
    pickScenariosBySet(scenario_set) {
        /** 
         * Get all the scenarios, that belong to a given scenario_set
         * and set an 'active' flag for those that are configured in the
         * SosModelRun
         * 
         * Arguments
         * ---------
         * scenario_set: str
         *     Name identifier of the scenario_set
         * 
         * Returns
         * -------
         * Object
         *     All scenarios that belong to the given scenario_set
         */ 

        let scenarios_in_sets = new Object()

        for (var i = 0; i < scenario_set.length; i++) {

            // Get all scenarios that belong to this scenario set
            scenarios_in_sets[scenario_set[i]] = this.props.scenarios.filter(scenario => scenario.scenario_set === scenario_set[i])

            // Flag the ones that are active in the modelrun configuration
            for (var k = 0; k < scenarios_in_sets[scenario_set[i]].length; k++) {

                scenarios_in_sets[scenario_set[i]][k].active = false

                if (this.props.sos_model_run.scenarios != null) {
                    this.props.sos_model_run.scenarios.forEach(function(element) {
                        if (scenarios_in_sets[scenario_set[i]][k].name == element[scenarios_in_sets[scenario_set[i]][k].scenario_set]) {
                            scenarios_in_sets[scenario_set[i]][k].active = true
                        }
                    })
                }              
            }
        }
        return scenarios_in_sets;
    }

    pickNarrativesBySet(narrative_set) {
        let narratives_in_sets = new Object()

        for (var i = 0; i < narrative_set.length; i++) {

            // Get all narratives that belong to this narrative set
            narratives_in_sets[narrative_set[i]] = this.props.narratives.filter(narrative => narrative.narrative_set === narrative_set[i])

            // Flag the ones that are active in the modelrun configuration
            for (var k = 0; k < narratives_in_sets[narrative_set[i]].length; k++) {

                narratives_in_sets[narrative_set[i]][k].active = false

                if (this.props.sos_model_run.narratives != null) {
                    this.props.sos_model_run.narratives.forEach(function(narratives) {
                        if (narratives[narratives_in_sets[narrative_set[i]][k].narrative_set] != null) {
                            narratives[narratives_in_sets[narrative_set[i]][k].narrative_set].forEach(function(narrative) {
                                if (narratives_in_sets[narrative_set[i]][k].name == narrative) {
                                    narratives_in_sets[narrative_set[i]][k].active = true
                                }
                            })
                        }
                    })
                }
            }           
        }
        return narratives_in_sets
    }

    selectSosModel(event) {
        let sos_model = this.pickSosModelByName(event.target.value)
        this.setState({selectedSosModel: sos_model})

        let scenarios = this.pickScenariosBySet(sos_model.scenario_sets)
        this.setState({selectedScenarios: scenarios})

        let narratives = this.pickNarrativesBySet(sos_model.narrative_sets)
        this.setState({selectedNarratives: narratives})
    }

    handleScenariosChange(scenario_set, scenario) {
    /**
     * Set a scenario change in the local selectedSosModelRun state representation
     * 
     * Arguments
     * ---------
     * scenario_set: str
     *     The scenario set that has changed
     * scenario: str
     *     The the scenario that has been selected
     */
        const { scenarios } = this.state.selectedSosModelRun
        
        if (scenarios === undefined) {
            // there are no scenarios defined
            // Initialize array
            // Add scenario_set and scenario
            let obj = {}
            obj[scenario_set] = [scenario]

            this.setState({
                selectedSosModelRun: update(this.state.selectedSosModelRun, {scenarios: {$set: [obj]}})
            })
        }
        else {
            for (let i = 0; i < scenarios.length; i++) {
                if (scenarios[i][scenario_set] != null) {
                    scenarios[i][scenario_set] = scenario
                }
            }
        }
    }

    handleNarrativeChange(narrative_set, narrative, active) {
    /**
     * Add or remove a narrative from the local selectedSosModelRun state representation
     * 
     * Arguments
     * ---------
     * narrative_set:
     *     The narrative set that has been changed
     * narrative:
     *     The narrative that has been changed
     * active:
     *     The new state of this narrative
     */
        const {narratives} = this.state.selectedSosModelRun

        if (narratives === undefined) {
            // there are no narratives defined
            // Initialize array
            // Add narrative_set and narrative
            let obj = {}
            obj[narrative_set] = [narrative]

            this.setState({
                selectedSosModelRun: update(this.state.selectedSosModelRun, {narratives: {$set: [obj]}})
            })
        }
        else {
            // there are already narratives defined
            for (let i = 0; i < narratives.length; i++) {

                if (Object.keys(narratives[i]) == narrative_set) {

                    for (let k = 0; k <= narratives[i][narrative_set].length; k++) {

                        if ((narratives[i][narrative_set][k] == narrative) && !active) {
                            // Remove narrative to set                      
                            narratives[i][narrative_set].splice(k, 1)

                            // If there are no narratives left in this set, remove the set
                            if (narratives[i][narrative_set].length == 0) {
                                narratives.splice(i, 1)
                            }

                            // If there are no narrative sets left, remove the narratives
                            console.log(narratives.length)
                            if (narratives.length == 0) {
                                this.setState({
                                    selectedSosModelRun: update(this.state.selectedSosModelRun, {narratives: {$set: undefined }})
                                })
                            }
                            break;
                        }
                        if (typeof narratives[i][narrative_set][k] === 'undefined') {
                            // Add narrative to set
                            narratives[i][narrative_set].push(narrative)
                            break;
                        }
                    }
                    break;
                }
                if (i == (narratives.length - 1)) {
                    // Narrative set does not exist in ModelrunConfig
                    // Add narrative_set and narrative
                    let obj = {}
                    obj[narrative_set] = [narrative]

                    narratives.push(obj)
                    break;
                }
            }
        }
    }

    handleTimestepChange(timesteps) {
        console.log(timesteps)

        this.setState({
            selectedSosModelRun: update(this.state.selectedSosModelRun, {timesteps: {$set: timesteps}})
        })
    }

    handleInputChange(event) {

        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedSosModelRun: update(this.state.selectedSosModelRun, {[name]: {$set: value}})
        })
    }

    handleSave(event) {
        this.props.save_model_run(this.state.selectedSosModelRun)
    }

    render() {
        const { sos_model_run, sos_models, scenarios, narratives } = this.props

        return (
            <div>

                <h3>General</h3>
                <label>Name:</label>
                <input name="name" type="text" defaultValue={sos_model_run.name} onChange={this.handleInputChange}/>

                <label>Description:</label>
                <div className="textarea-container">
                    <textarea name="description" rows="5" defaultValue={sos_model_run.description} onChange={this.handleInputChange}/>
                </div>

                <label>Datestamp:</label>
                <input type="datetime-local" defaultValue=""/>

                <h3>Model</h3>
                <label>System-of-systems model:</label>
                <div className="select-container">
                    <select name="sos_model" type="select" value={this.state.selectedSosModel.name} onChange={(event) => {this.selectSosModel(event); this.handleInputChange(event);}}>
                        <option disabled="disabled" >Please select a system-of-systems model</option>
                        {
                            sos_models.map((sos_model) => (
                                <option key={sos_model.name} value={sos_model.name}>{sos_model.name}</option>
                            ))
                        }
                    </select>
                </div>

                <h3>Scenarios</h3>
                <fieldset>            
                    {         
                        Object.keys(this.state.selectedScenarios).map((item, i) =>
                            <ScenarioSelector key={i} scenarioSet={item} scenarios={this.state.selectedScenarios[item]} change_scenario={this.handleScenariosChange} />
                        )
                    }
                </fieldset>

                <h3>Narratives</h3>
                <fieldset>            
                    {         
                        Object.keys(this.state.selectedNarratives).map((item, i) =>
                            <NarrativeSelector key={i} narrativeSet={item} narratives={this.state.selectedNarratives[item]} change_narrative={this.handleNarrativeChange}/>
                        )
                    }
                </fieldset>

                <h3>Timesteps</h3>
                <TimestepSelector defaultValue={sos_model_run.timesteps} onChange={this.handleTimestepChange}/>

                <input type="button" value="Save Model Run Configuration" onClick={this.handleSave} />
            </div>
        );
    }
}

SosModelRunConfigForm.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    save_model_run: PropTypes.func.isRequired
};

export default SosModelRunConfigForm;
