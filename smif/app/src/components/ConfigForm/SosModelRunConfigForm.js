import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import SosModelSelector from './SosModelRun/SosModelSelector.js'
import ScenarioSelector from './SosModelRun/ScenarioSelector.js'
import NarrativeSelector from './SosModelRun/NarrativeSelector.js'
import TimestepSelector from './SosModelRun/TimestepSelector.js'

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleSosModelChange = this.handleSosModelChange.bind(this)
        this.handleScenariosChange = this.handleScenariosChange.bind(this)
        this.handleNarrativeChange = this.handleNarrativeChange.bind(this)
        this.handleTimestepChange = this.handleTimestepChange.bind(this)
        this.handleInputChange = this.handleInputChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedSosModelRun = this.props.sosModelRun
    }

    componentDidMount(){
        if (Object.keys(document).length) {
            document.addEventListener("keydown", this.handleKeyPress, false)
        }
    }

    componentWillUnmount(){
        document.removeEventListener("keydown", this.handleKeyPress, false)
    }

    handleKeyPress(){
        if(event.keyCode === 27) {
            this.handleCancel()
        }
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

        scenarios[scenario_set] = scenario
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

        if (narratives === undefined || narratives[0] === undefined) {
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
        this.props.saveModelRun(this.state.selectedSosModelRun)
    }

    handleCancel() {
        this.props.cancelModelRun()
    }

    render() {
        const {sosModels, scenarios, narratives} = this.props
        const {selectedSosModelRun} = this.state

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input id="sos_model_run_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedSosModelRun.name} onChange={this.handleInputChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea id="sos_model_run_description" className="form-control" name="description" rows="5" defaultValue={selectedSosModelRun.description} onChange={this.handleInputChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Created</label>
                            <div className="col-sm-10">
                                <label id="sos_model_run_stamp" className="form-control">{selectedSosModelRun.stamp}</label>
                            </div>
                        </div>
                    </div>
                </div>

                <br/>

                <div className="card">
                    <div className="card-header">Model</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">System-of-systems model</label>
                            <div className="col-sm-10">
                                <SosModelSelector id="sos_model_run_sos_model" sosModelRun={selectedSosModelRun} sosModels={sosModels} onChange={this.handleSosModelChange} />
                            </div>
                        </div>

                    </div>
                </div>

                <br/>

                <div className="card">
                    <div className="card-header">Settings</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Scenarios</label>
                            <div className="col-sm-10">
                                <ScenarioSelector sosModelRun={selectedSosModelRun} sosModels={sosModels} scenarios={scenarios} onChange={this.handleScenariosChange} />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Narratives</label>
                            <div className="col-sm-10">
                                <NarrativeSelector sosModelRun={selectedSosModelRun} sosModels={sosModels} narratives={narratives} onChange={this.handleNarrativeChange} />
                            </div>
                        </div>
                    </div>
                </div>

                <br/>

                <div className="card">
                    <div className="card-header">Timesteps</div>
                    <div className="card-body">
                        <TimestepSelector timeSteps={selectedSosModelRun.timesteps} onChange={this.handleTimestepChange}/>
                    </div>
                </div>

                <br/>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save Model Run Configuration" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SosModelRunConfigForm.propTypes = {
    sosModelRun: PropTypes.object.isRequired,
    sosModels: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    saveModelRun: PropTypes.func,
    cancelModelRun: PropTypes.func
}

export default SosModelRunConfigForm
