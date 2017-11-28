import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Link } from 'react-router-dom'
import update from 'react-addons-update'

import { saveSosModelRun } from '../actions/actions.js'

import SosModelSelector from '../components/SosModelRunConfigForm/SosModelSelector.js'
import ScenarioSelector from '../components/SosModelRunConfigForm/ScenarioSelector.js'
import NarrativeSelector from '../components/SosModelRunConfigForm/NarrativeSelector.js'
import TimestepSelector from '../components/SosModelRunConfigForm/TimestepSelector.js'

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleSosModelChange = this.handleSosModelChange.bind(this)
        this.handleScenariosChange = this.handleScenariosChange.bind(this)
        this.handleNarrativeChange = this.handleNarrativeChange.bind(this)
        this.handleTimestepChange = this.handleTimestepChange.bind(this)
        this.handleInputChange = this.handleInputChange.bind(this)
        this.handleSave = this.handleSave.bind(this)

        this.state = {}
        this.state.selectedSosModelRun = this.props.sosModelRun
    }

    handleSosModelChange(sos_model) {
        this.setState({
            selectedSosModelRun: update(this.state.selectedSosModelRun, {sos_model: {$set: sos_model}})
        })
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

        if (scenarios === undefined || scenarios[0] === undefined) {
            // there are no scenarios defined
            // Initialize array
            // Add scenario_set and scenario
            scenarios.push({[scenario_set]: scenario})
        }
        else {
            for (let i = 0; i < scenarios.length; i++) {
                if (scenarios[i][scenario_set] != null) {
                    scenarios[i][scenario_set] = scenario
                    break
                } else if (i == (scenarios.length - 1)) {
                    scenarios.push({[scenario_set]: scenario})
                    break
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
                            if (narratives.length == 0) {
                                this.setState({
                                    selectedSosModelRun: update(this.state.selectedSosModelRun, {narratives: {$set: undefined }})
                                })
                            }
                            break
                        }
                        
                        if (typeof narratives[i][narrative_set][k] === 'undefined') {
                            // Add narrative to set
                            narratives[i][narrative_set].push(narrative)
                            break
                        }
                    }
                    break
                }
                if (i == (narratives.length - 1)) {
                    // Narrative set does not exist in ModelrunConfig
                    // Add narrative_set and narrative
                    let obj = {}
                    obj[narrative_set] = [narrative]

                    narratives.push(obj)
                    break
                }
            }
        }
    }

    handleTimestepChange(timesteps) {
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

    handleSave() {
        console.log(this.state)
        this.props.saveModelRun(this.state.selectedSosModelRun)
    }

    render() {
        const {sosModels, scenarios, narratives} = this.props
        const {selectedSosModelRun} = this.state

        return (
            <div>
                <h3>General</h3>
                <label>Name:</label>
                <input name="name" type="text" disabled="true" defaultValue={selectedSosModelRun.name} onChange={this.handleInputChange}/>

                <label>Description:</label>
                <div className="textarea-container">
                    <textarea name="description" rows="5" defaultValue={selectedSosModelRun.description} onChange={this.handleInputChange}/>
                </div>

                <label>Created: {selectedSosModelRun.stamp}</label>
                

                <h3>Model</h3>
                <label>System-of-systems model:</label>
                <SosModelSelector sosModelRun={selectedSosModelRun} sosModels={sosModels} onChange={this.handleSosModelChange} />

                <h3>Scenarios</h3>
                <ScenarioSelector sosModelRun={selectedSosModelRun} sosModels={sosModels} scenarios={scenarios} onChange={this.handleScenariosChange} />

                <h3>Narratives</h3>
                <NarrativeSelector sosModelRun={selectedSosModelRun} sosModels={sosModels} narratives={narratives} onChange={this.handleNarrativeChange} />

                <h3>Timesteps</h3>
                <TimestepSelector defaultValue={selectedSosModelRun.timesteps} onChange={this.handleTimestepChange}/>

                <input type="button" value="Save Model Run Configuration" onClick={this.handleSave} />
            </div>
        )
    }
}

SosModelRunConfigForm.propTypes = {
    sosModelRun: PropTypes.object.isRequired,
    sosModels: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    saveModelRun: PropTypes.func.isRequired
}

export default SosModelRunConfigForm
