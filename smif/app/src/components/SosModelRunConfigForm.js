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
        this.state.selectedSosModelRun = this.props.sosModelRun
        this.state.selectedSosModel = this.pickSosModelByName(this.props.sosModelRun.sos_model)
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

        let sos_model = this.props.sosModels.filter(
            (sos_model) => sos_model.name === sos_model_name
        )[0]

        if (typeof sos_model === 'undefined') {
            sos_model = this.props.sosModels[0]
        }
        
        return sos_model
    }

    selectSosModel(event) {
        let sos_model = this.pickSosModelByName(event.target.value)
        this.setState({selectedSosModel: sos_model})
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
        console.log(scenario_set)
        console.log(scenario)

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
        this.props.saveModelRun(this.state.selectedSosModelRun)
        console.log(this.state)
    }

    render() {
        const { sosModelRun, sosModels, scenarios, narratives } = this.props

        return (
            <div>

                <h3>General</h3>
                <label>Name:</label>
                <input name="name" type="text" defaultValue={sosModelRun.name} onChange={this.handleInputChange}/>

                <label>Description:</label>
                <div className="textarea-container">
                    <textarea name="description" rows="5" defaultValue={sosModelRun.description} onChange={this.handleInputChange}/>
                </div>

                <label>Datestamp:</label>
                <input type="datetime-local" defaultValue=""/>

                <h3>Model</h3>
                <label>System-of-systems model:</label>
                <div className="select-container">
                    <select name="sos_model" type="select" value={this.state.selectedSosModel.name} onChange={(event) => {this.selectSosModel(event); this.handleInputChange(event);}}>
                        <option disabled="disabled" >Please select a system-of-systems model</option>
                        {
                            sosModels.map((sosModel) => (
                                <option key={sosModel.name} value={sosModel.name}>{sosModel.name}</option>
                            ))
                        }
                    </select>
                </div>

                <h3>Scenarios</h3>
                <ScenarioSelector sosModelRun={sosModelRun} sosModels={sosModels} scenarios={scenarios} onChange={this.handleScenariosChange} />

                <h3>Narratives</h3>
                <NarrativeSelector sosModelRun={sosModelRun} sosModels={sosModels} narratives={narratives} onChange={this.handleNarrativeChange} />

                <h3>Timesteps</h3>
                <TimestepSelector defaultValue={sosModelRun.timesteps} onChange={this.handleTimestepChange}/>

                <input type="button" value="Save Model Run Configuration" onClick={this.handleSave} />
            </div>
        );
    }
}

SosModelRunConfigForm.propTypes = {
    sosModelRun: PropTypes.object.isRequired,
    sosModels: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    saveModelRun: PropTypes.func.isRequired
};

export default SosModelRunConfigForm;
