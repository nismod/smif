import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertySelector from 'components/ConfigForm/General/PropertySelector.js'
import DependencyList from 'components/ConfigForm/SosModel/DependencyList.js'
import NarrativeList from 'components/ConfigForm/SosModel/NarrativeList.js'

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

    handleChange(key, value) {
        this.props.onEdit()

        this.setState({
            selectedSosModel: update(this.state.selectedSosModel, {[key]: {$set: value}})
        })
    }

    handleSave() {
        this.props.onSave(this.state.selectedSosModel)
        this.props.onCancel()
    }

    handleCancel() {
        this.props.onCancel()
    }

    render() {
        const {selectedSosModel} = this.state

        if (this.props.save) {
            this.props.onSave(this.state.selectedSosModel)
        }

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
                                    rows="5" 
                                    defaultValue={selectedSosModel.description} 
                                    onChange={(event) => this.handleChange('description', event.target.value)}/>
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
                                    activeProperties={selectedSosModel.sector_models} 
                                    availableProperties={this.props.sector_models} 
                                    onChange={(event) => this.handleChange('sector_models', event.target.value)} />
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
                                    activeProperties={selectedSosModel.scenarios} 
                                    availableProperties={this.props.scenarios} 
                                    onChange={(event) => this.handleChange('scenarios', event.target.value)} />
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
                            onChange={(model_dependencies) => this.handleChange('model_dependencies', model_dependencies)} 
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
                            onChange={(scenario_dependencies) => this.handleChange('scenario_dependencies', scenario_dependencies)} 
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

                <div className="card">
                    <div className="card-header">Narratives</div>
                    <div className="card-body">
                        <NarrativeList 
                            name="Narrative" 
                            narratives={selectedSosModel.narratives}
                            sector_models={this.props.sector_models.filter(
                                sector_model => this.state.selectedSosModel.sector_models.includes(sector_model.name)
                            )}
                            onChange={(narratives) => this.handleChange('narratives', narratives)}  />
                        {   
                            'narratives' in errors
                                ? (
                                    <div className="invalid-feedback">
                                        {
                                            errors['narratives'].map((exception, idx) => (
                                                <div key={'feedback_narratives_' + idx}>
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
    save: PropTypes.bool,
    onSave: PropTypes.func,
    onCancel: PropTypes.func,
    onEdit: PropTypes.func
}

export default SosModelConfigForm
