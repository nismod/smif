import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertySelector from '../components/PropertySelector.js'
import InputsOutputsForm from '../components/SectorModelConfigForm/InputsOutputsForm.js'
import ParameterSelector from '../components/SectorModelConfigForm/ParameterSelector.js'
import PropertyList from '../components/PropertyList.js'

class SectorModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedSectorModel = this.props.sectorModel
        
        this.handleChange = this.handleChange.bind(this)
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedSectorModel: update(this.state.selectedSectorModel, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveSectorModel(this.state.selectedSectorModel)
    }

    handleCancel() {
        this.props.cancelSectorModel()
    }

    render() {
        const {sectorModel, sectorModels, scenarioSets, scenarios, narrativeSets, narratives} = this.props
        const {selectedSectorModel} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="name" type="text" disabled="true" defaultValue={selectedSectorModel.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea className="form-control" name="description" rows="5" defaultValue={selectedSectorModel.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Environment</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Class Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="classname" type="text" defaultValue={selectedSectorModel.classname} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Path</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="path" type="text" defaultValue={selectedSectorModel.path} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Inputs</div>
                        <div className="card-body">
                            <PropertyList itemsName="inputs" items={selectedSectorModel.inputs} columns={{name: 'Name', spatial_resolution: 'Spatial Resolution', temporal_resolution: 'Temporal Resolution', units: 'Units'}} editButton={false} deleteButton={true} onEdit="" onDelete={this.handleChange} />
                            <InputsOutputsForm items={selectedSectorModel.inputs} isInputs={true} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Outputs</div>
                        <div className="card-body">
                            <PropertyList itemsName="outputs" items={selectedSectorModel.outputs} columns={{name: 'Name', spatial_resolution: 'Spatial Resolution', temporal_resolution: 'Temporal Resolution', units: 'Units'}} editButton={false} deleteButton={true} onEdit="" onDelete={this.handleChange} />
                            <InputsOutputsForm items={selectedSectorModel.outputs} isOutputs={true} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Parameters</div>
                        <div className="card-body">
                            <PropertyList itemsName="parameters" items={selectedSectorModel.parameters} columns={{name: 'Name', description: 'Description', default_value: 'Default Value', units: 'Units', absolute_range: 'Absolute Range', suggested_range: 'Suggested Range'}} editButton={false} deleteButton={true} onEdit="" onDelete={this.handleChange} />
                            <ParameterSelector parameters={selectedSectorModel.parameters} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    {/* <div className="card">
                        <div className="card-header">Settings</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Sector Models</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="sector_models" activeProperties={selectedSectorModel.sector_models} availableProperties={sectorModels} onChange={this.handleChange} />
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Scenario Sets</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="scenario_sets" activeProperties={selectedSectorModel.scenario_sets} availableProperties={scenarioSets} onChange={this.handleChange} />
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Narrative Sets</label>
                                <div className="col-sm-10">
                                    <PropertySelector name="narrative_sets" activeProperties={selectedSectorModel.narrative_sets} availableProperties={narrativeSets} onChange={this.handleChange} />
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Iterations</label>
                                <div className="col-sm-10">
                                    <div className="input-group">
                                        <span className="input-group-addon">Maximum</span>
                                        <input className="form-control" name="max_iterations" type="number" min="1" defaultValue={selectedSectorModel.max_iterations} onChange={this.handleChange}/>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Dependencies</div>
                        <div className="card-body">
                            <PropertyList itemsName="dependencies" items={selectedSectorModel.dependencies} columns={['Source Model', 'Output', 'Sink Model', 'Input']} editButton={false} deleteButton={true} onEdit="" onDelete={this.handleChange} />
                            <DependencySelector dependencies={selectedSectorModel.dependencies} sectorModels={sectorModels} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Convergence Tolerance</div>
                        <div className="card-body">
                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Absolute</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="convergence_absolute_tolerance" type="number" step="0.00000001" min="0.00000001" defaultValue={selectedSectorModel.convergence_absolute_tolerance} onChange={this.handleChange}/>
                                </div>
                            </div>
                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Relative</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="convergence_relative_tolerance" type="number" step="0.00000001" min="0.00000001" defaultValue={selectedSectorModel.convergence_relative_tolerance} onChange={this.handleChange}/>
                                </div>
                            </div>
                        </div>
                    </div> */}

                    <br/>
                </form>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save Sector Model Configuration" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SectorModelConfigForm.propTypes = {
    sectorModel: PropTypes.object.isRequired,
    sectorModels: PropTypes.array.isRequired,
    scenarioSets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrativeSets: PropTypes.array.isRequired,
    saveSectorModel: PropTypes.func.isRequired,
    cancelSectorModel: PropTypes.func.isRequired
}

export default SectorModelConfigForm