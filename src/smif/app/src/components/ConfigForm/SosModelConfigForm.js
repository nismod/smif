import React, { Component } from 'react'
import PropTypes, { object } from 'prop-types'
import update from 'immutability-helper'

import PropertySelector from './General/PropertySelector.js'
import DependencySelector from './SosModel/DependencySelector.js'
import PropertyList from './General/PropertyList.js'

class SosModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedSosModel = this.props.sosModel
        this.state.dependencyWarning = []
    }

    componentDidMount(){
        document.addEventListener("keydown", this.handleKeyPress, false)
        this.validateForm()
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

        this.validateForm()

        this.setState({
            selectedSosModel: update(this.state.selectedSosModel, {[name]: {$set: value}})
        })
    }

    validateForm() {
        const {selectedSosModel} = this.state

        if (Object.keys(selectedSosModel).length > 0) {

            // Check for dependencies that contain sector models or scenario sets that are not configured
            let illegalDependencies = []

            for (let i = 0; i < selectedSosModel.dependencies.length; i++) {
                if ((selectedSosModel.sector_models.includes(selectedSosModel.dependencies[i].sink_model) || selectedSosModel.scenario_sets.includes(selectedSosModel.dependencies[i].sink_model)) &&
                    (selectedSosModel.sector_models.includes(selectedSosModel.dependencies[i].source_model) || selectedSosModel.scenario_sets.includes(selectedSosModel.dependencies[i].source_model))) {
                    illegalDependencies[i] = false
                } else {
                    illegalDependencies[i] = true
                }
            }

            this.setState({
                dependencyWarning: illegalDependencies
            })
        }
        
    }

    handleSave() {
        this.props.saveSosModel(this.state.selectedSosModel)
    }

    handleCancel() {
        this.props.cancelSosModel()
    }

    render() {
        const {sectorModels, scenarioSets, narrativeSets} = this.props
        const {selectedSosModel} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="sos_model_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedSosModel.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="sos_model_description" className="form-control" name="description" rows="5" defaultValue={selectedSosModel.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Settings</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Sector Models</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="sector_models" activeProperties={selectedSosModel.sector_models} availableProperties={sectorModels} onChange={this.handleChange} />
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Scenario Sets</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="scenario_sets" activeProperties={selectedSosModel.scenario_sets} availableProperties={scenarioSets} onChange={this.handleChange} />
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Narrative Sets</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="narrative_sets" activeProperties={selectedSosModel.narrative_sets} availableProperties={narrativeSets} onChange={this.handleChange} />
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Dependencies</div>
                        <div className="card-body">
                            <PropertyList itemsName="dependencies" items={selectedSosModel.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} enableWarnings={true} rowWarning={this.state.dependencyWarning} editButton={false} deleteButton={true} onDelete={this.handleChange} />
                            <DependencySelector dependencies={selectedSosModel.dependencies} sectorModels={selectedSosModel.sector_models} scenarioSets={selectedSosModel.scenario_sets} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Iteration Settings</div>
                        <div className="card-body">
                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Maximum Iterations</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="max_iterations" type="number" min="1" defaultValue={selectedSosModel.max_iterations} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Absolute Convergence Tolerance</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="convergence_absolute_tolerance" type="number" step="0.00000001" min="0.00000001" defaultValue={selectedSosModel.convergence_absolute_tolerance} onChange={this.handleChange}/>
                                </div>
                            </div>
                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Relative Convergence Tolerance</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="convergence_relative_tolerance" type="number" step="0.00000001" min="0.00000001" defaultValue={selectedSosModel.convergence_relative_tolerance} onChange={this.handleChange}/>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>
                </form>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save System-of-systems Model Configuration" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SosModelConfigForm.propTypes = {
    sosModel: PropTypes.object.isRequired,
    sectorModels: PropTypes.array.isRequired,
    scenarioSets: PropTypes.array.isRequired,
    narrativeSets: PropTypes.array.isRequired,
    saveSosModel: PropTypes.func,
    cancelSosModel: PropTypes.func
}

export default SosModelConfigForm
