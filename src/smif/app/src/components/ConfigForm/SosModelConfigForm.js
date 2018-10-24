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

        let errors = {}
        if (this.props.error.SmifDataInputError != undefined) {
            errors = this.props.error.SmifDataInputError.reduce(function(map, obj) {
                if (!(obj.component in map)) {
                    map[obj.component] = []
                }
                map[obj.component].push({
                    'error': obj.error,
                    'message': obj.message
                })
                return map
            }, {})
        }
        
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
                                <textarea id="sos_model_description" 
                                    className={
                                        'description' in errors
                                            ? 'form-control is-invalid'   
                                            : 'form-control'
                                    }
                                    name="description" 
                                    rows="5" 
                                    defaultValue={selectedSosModel.description} 
                                    onChange={this.handleChange}/>
                                
                                {   
                                    'description' in errors
                                        ? (
                                            <div className="invalid-feedback">
                                                {
                                                    errors['description'].map((exception, idx) => (
                                                        <div key={'feedback_description_' + idx}>
                                                            {exception.error + ' ' + exception.message}
                                                        </div>
                                                        
                                                    ))
                                                }
                                            </div>)
                                        : ''
                                }
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
                                {   
                                    'sector_models' in errors
                                        ? (
                                            <div className="invalid-feedback">
                                                {
                                                    errors['sector_models'].map((exception, idx) => (
                                                        <div key={'feedback_sector_models_' + idx}>
                                                            {exception.error + ' ' + exception.message}
                                                        </div>
                                                        
                                                    ))
                                                }
                                            </div>)
                                        : ''
                                }
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Scenarios</label>
                            <div className="col-sm-10">
                                <PropertySelector 
                                    name="scenarios" 
                                    activeProperties={selectedSosModel.scenarios} 
                                    availableProperties={this.props.scenarios} 
                                    onChange={this.handleChange} />
                                {   
                                    'scenarios' in errors
                                        ? (
                                            <div className="invalid-feedback">
                                                {
                                                    errors['scenarios'].map((exception, idx) => (
                                                        <div key={'feedback_scenarios_' + idx}>
                                                            {exception.error + ' ' + exception.message}
                                                        </div>
                                                        
                                                    ))
                                                }
                                            </div>)
                                        : ''
                                }
                            </div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Model Dependencies</div>
                    <div className="card-body">
  
                        <DependencyList 
                            name="Model Dependency" 
                            dependencies={this.state.selectedSosModel.model_dependencies} 
                            source={
                                this.props.sector_models.filter(
                                    sector_model => this.state.selectedSosModel.sector_models.includes(sector_model.name)
                                )
                            }
                            source_output={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.outputs
                                    return obj}, {}
                                )
                            }
                            sink={
                                this.props.sector_models.filter(
                                    sector_model => this.state.selectedSosModel.sector_models.includes(sector_model.name)
                                )
                            }
                            sink_input={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.inputs
                                    return obj}, {}
                                )
                            } 
                        />
                        {   
                            'model_dependencies' in errors
                                ? (
                                    <div className="invalid-feedback">
                                        {
                                            errors['model_dependencies'].map((exception, idx) => (
                                                <div key={'feedback_model_dependencies_' + idx}>
                                                    {exception.error + ' ' + exception.message}
                                                </div>
                                                
                                            ))
                                        }
                                    </div>)
                                : ''
                        }
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Scenario Dependencies</div>
                    <div className="card-body">

                        <DependencyList 
                            name="Scenario Dependency" 
                            dependencies={this.state.selectedSosModel.scenario_dependencies} 
                            source={
                                this.props.scenarios.filter(
                                    scenario => this.state.selectedSosModel.scenarios.includes(scenario.name)
                                )
                            } 
                            source_output={
                                this.props.scenarios.reduce(function(obj, item) {
                                    obj[item.name] = item.provides
                                    return obj}, {}
                                )
                            }
                            sink={
                                this.props.sector_models.filter(
                                    sector_model => this.state.selectedSosModel.sector_models.includes(sector_model.name)
                                )
                            }
                            sink_input={
                                this.props.sector_models.reduce(function(obj, item) {
                                    obj[item.name] = item.inputs
                                    return obj}, {}
                                )
                            } 
                        />
                        {   
                            'scenario_dependencies' in errors
                                ? (
                                    <div className="invalid-feedback">
                                        {
                                            errors['scenario_dependencies'].map((exception, idx) => (
                                                <div key={'feedback_scenario_dependencies_' + idx}>
                                                    {exception.error + ' ' + exception.message}
                                                </div>
                                                
                                            ))
                                        }
                                    </div>)
                                : ''
                        }
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
    error: PropTypes.object.isRequired,
    saveSosModel: PropTypes.func,
    cancelSosModel: PropTypes.func
}

export default SosModelConfigForm
