import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertyList from './General/PropertyList'
import ParameterFileSelector from './Scenario/ParameterFileSelector'

class ScenarioConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedScenario = this.props.scenario
    }

    componentDidMount(){
        document.addEventListener("keydown", this.handleKeyPress, false)
    }

    componentWillUnmount(){
        document.removeEventListener("keydown", this.handleKeyPress, false)
    }

    handleKeyPress(){
        if(event.keyCode === 27) {
            this.handleCancel()
        }
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedScenario: update(this.state.selectedScenario, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveScenario(this.state.selectedScenario)
    }

    handleCancel() {
        this.props.cancelScenario()
    }

    render() {
        const {scenarioSets} = this.props
        const {selectedScenario} = this.state

        let selectedScenarioSet = {name: '', description: ''}
        let scenarioSetSelected = false

        if (selectedScenario.scenario_set != '') {
            selectedScenarioSet = scenarioSets.filter(scenarioSet => scenarioSet.name == selectedScenario.scenario_set)[0]
            scenarioSetSelected = true
        }

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="scenario_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedScenario.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="scenario_description" className="form-control" name="description" rows="5" defaultValue={selectedScenario.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Settings</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Scenario Set</label>
                                <div className="col-sm-10">
                                    <select className="form-control" name="scenario_set" defaultValue={selectedScenario.scenario_set} onChange={this.handleChange}>
                                        <option disabled="disabled" value="" >Please select a Scenario Set</option>
                                        {
                                            scenarioSets.map(scenarioSet =>
                                                <option key={scenarioSet.name} value={scenarioSet.name}>{scenarioSet.name}</option>
                                            )
                                        }
                                    </select>
                                    <br/>
                                    <div className="alert alert-dark" hidden={!scenarioSetSelected} role="alert">
                                        {selectedScenarioSet && selectedScenarioSet.description}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Parameters</div>
                        <div className="card-body">
                            <PropertyList itemsName="parameters" items={selectedScenario.parameters} columns={{name: 'Name', filename: 'Filename', spatial_resolution: 'Spatial Resolution', temporal_resolution: 'Temporal Resolution', units: 'Units'}} editButton={false} deleteButton={true} onDelete={this.handleChange} />
                            <ParameterFileSelector parameters={selectedScenario.parameters} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                </form>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save Sector Model Configuration" onClick={this.handleSave} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ScenarioConfigForm.propTypes = {
    scenario: PropTypes.object.isRequired,
    scenarioSets: PropTypes.array.isRequired,
    saveScenario: PropTypes.func,
    cancelScenario: PropTypes.func
}

export default ScenarioConfigForm
