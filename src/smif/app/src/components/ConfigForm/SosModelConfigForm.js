import React, { Component } from 'react'
import PropTypes, { object } from 'prop-types'
import update from 'immutability-helper'

import Popup from './General/Popup.js'
import PropertySelector from './General/PropertySelector.js'
import DependencySelector from './SosModel/DependencySelector.js'
import PropertyList from './General/PropertyList.js'
import DeleteForm from '../../components/ConfigForm/General/DeleteForm.js'

class SosModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selectedSosModel: this.props.sosModel,
            dependencyWarning: [],
            deletePopupIsOpen: false
        }

        this.closeDeletePopup = this.closeDeletePopup.bind(this)
        this.openDeletePopup = this.openDeletePopup.bind(this)
        this.handleDelete = this.handleDelete.bind(this)
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

    handleDelete(config) {
        const {deletePopupType, selectedSosModel} = this.state

        switch(deletePopupType) {
            case 'dependency':
                this.state.selectedSosModel.dependencies.splice(config, 1)
        }

        this.forceUpdate()
        this.closeDeletePopup()
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

    openDeletePopup(event) {
        this.setState({
            deletePopupIsOpen: true,
            deletePopupConfigName: event.target.value,
            deletePopupType: event.target.name,
        })
    }

    closeDeletePopup() {
        this.setState({deletePopupIsOpen: false})
    }

    render() {
        const {sectorModels, scenarioSets, narrativeSets} = this.props
        const {selectedSosModel} = this.state

        // Get dependencies, give an index as name
        let dependencies = []
        if (selectedSosModel.dependencies != undefined) {
            for (let i = 0; i < selectedSosModel.dependencies.length; i++) {
                dependencies.push({
                    name: i,
                    sink_model: selectedSosModel.dependencies[i].sink_model,
                    sink_model_input: selectedSosModel.dependencies[i].sink_model_input,
                    source_model: selectedSosModel.dependencies[i].source_model,
                    source_model_output: selectedSosModel.dependencies[i].source_model_output
                })
            }
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
                            <PropertyList itemsName="dependency" items={dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} enableWarnings={true} rowWarning={this.state.dependencyWarning} editButton={false} deleteButton={true} onDelete={this.openDeletePopup} />
                            <DependencySelector sectorModels={sectorModels} scenarioSets={scenarioSets} dependencies={selectedSosModel.dependencies} selectedSectorModels={selectedSosModel.sector_models} selectedScenarioSets={selectedSosModel.scenario_sets} onChange={this.handleChange}/>
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

                <Popup onRequestOpen={this.state.deletePopupIsOpen}>
                    <DeleteForm config_name={this.state.deletePopupConfigName} config_type={this.state.deletePopupType} submit={this.handleDelete} cancel={this.closeDeletePopup}/>
                </Popup>


                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSave} />
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
