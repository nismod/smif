import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertySelector from 'components/ConfigForm/General/PropertySelector.js'
import DependencyList from 'components/ConfigForm/SosModel/DependencyList.js'
import { SaveButton, CancelButton } from 'components/ConfigForm/General/Buttons'

class SosModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selectedSosModel: this.props.sos_model
        }
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedSosModel: update(this.state.selectedSosModel, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveSosModel(this.state.selectedSosModel)
    }

    handleCancel() {
        this.props.cancelSosModel()
    }

    render() {
        const {selectedSosModel} = this.state

        return (
            <div>
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

                <div className="card">
                    <div className="card-header">Settings</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Sector Models</label>
                            <div className="col-sm-10">
                                <PropertySelector 
                                    name="sector_models" 
                                    activeProperties={selectedSosModel.sector_models} 
                                    availableProperties={this.props.sector_models} 
                                    onChange={this.handleChange} />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Scenarios</label>
                            <div className="col-sm-10">
                                <PropertySelector 
                                    name="scenario_sets" 
                                    activeProperties={selectedSosModel.scenarios} 
                                    availableProperties={this.props.scenarios} 
                                    onChange={this.handleChange} />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Narratives</label>
                            <div className="col-sm-10">
                                <PropertySelector 
                                    name="narrative_sets" 
                                    activeProperties={selectedSosModel.narratives} 
                                    availableProperties={this.props.narratives} 
                                    onChange={this.handleChange} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Scenario Dependencies</div>
                    <div className="card-body">
  
                        <DependencyList 
                            name="Scenario Dependency" 
                            dependencies={this.state.selectedSosModel.scenario_dependencies} 
                            source={this.props.scenarios} 
                            source_output={
                                this.props.scenarios.reduce(function(obj, item) {
                                    obj[item.name] = item.provides
                                    return obj}, {}
                                )
                            }
                            sink={this.props.sector_models}
                            sink_input={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.inputs
                                    return obj}, {}
                                )
                            } 
                        />
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Model Dependencies</div>
                    <div className="card-body">
  
                        <DependencyList 
                            name="Model Dependency" 
                            dependencies={this.state.selectedSosModel.model_dependencies} 
                            source={this.props.sector_models} 
                            source_output={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.outputs
                                    return obj}, {}
                                )
                            }
                            sink={this.props.sector_models}
                            sink_input={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.inputs
                                    return obj}, {}
                                )
                            } 
                        />
                    </div>
                </div>

                <SaveButton onClick={this.handleSave} />
                <CancelButton onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SosModelConfigForm.propTypes = {
    sos_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    saveSosModel: PropTypes.func,
    cancelSosModel: PropTypes.func
}

export default SosModelConfigForm
