import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertySelector from 'components/ConfigForm/General/PropertySelector.js'
import DependencyList from 'components/ConfigForm/SosModel/DependencyList.js'
import NarrativeList from 'components/ConfigForm/SosModel/NarrativeList.js'

import { PrimaryButton, SecondaryButton } from 'components/ConfigForm/General/Buttons'
import ErrorBlock from './General/ErrorBlock';

class SosModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selected: this.props.sos_model,
            inuse_dropdown: false
        }
    }

    handleChange(key, value) {
        this.props.onEdit()

        this.setState({
            selected: update(this.state.selected, {[key]: {$set: value}})
        })
    }

    handleSave() {
        this.props.onSave(this.state.selected)
        this.props.onCancel()
    }

    handleCancel() {
        this.props.onCancel()
    }

    render() {
        const {selected} = this.state

        if (this.props.save) {
            this.props.onSave(this.state.selected)
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
                                <input id="sos_model_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selected.name} onChange={this.handleChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <ErrorBlock errors={errors.description} />
                                <textarea id="sos_model_description"
                                    className={
                                        'description' in errors
                                            ? 'form-control is-invalid'
                                            : 'form-control'
                                    }
                                    rows="5"
                                    defaultValue={selected.description}
                                    onChange={(event) => this.handleChange('description', event.target.value)}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">In use by</label>
                            <div className="col-sm-10">
                                <div className="dropdown" onClick={() => this.setState({ inuse_dropdown: !this.state.inuse_dropdown})}>
                                    <button
                                        className="btn btn-secondary dropdown-toggle"
                                        type="button"
                                        id="dropdownMenuButton"
                                        data-toggle="dropdown"
                                        aria-haspopup="true"
                                    >
                                        Model Run Configuration
                                    </button>
                                    <div className={`dropdown-menu${this.state.inuse_dropdown ? ' show' : ''}`} aria-labelledby="dropdownMenuButton">
                                        {
                                            this.props.model_runs.filter(model_run => model_run.sos_model == this.props.sos_model.name).map(model_run => (
                                                <a key={model_run.name}
                                                    className="btn dropdown-item"
                                                    onClick={() => this.props.onNavigate('/configure/model-runs/' + model_run.name)}>
                                                    {model_run.name}
                                                </a>
                                            ))
                                        }
                                    </div>
                                </div>
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
                                <ErrorBlock errors={errors.sector_models} />
                                <PropertySelector
                                    activeProperties={selected.sector_models}
                                    availableProperties={this.props.sector_models}
                                    onChange={(event) => this.handleChange('sector_models', event.target.value)} />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Scenarios</label>
                            <div className="col-sm-10">
                                <ErrorBlock errors={errors.scenarios} />
                                <PropertySelector
                                    activeProperties={selected.scenarios}
                                    availableProperties={this.props.scenarios}
                                    onChange={(event) => this.handleChange('scenarios', event.target.value)} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Model Dependencies</div>
                    <div className="card-body">
                        <ErrorBlock errors={errors.model_dependencies} />
                        <DependencyList
                            name="Model Dependency"
                            dependencies={this.state.selected.model_dependencies}
                            source={
                                this.props.sector_models.filter(
                                    sector_model => this.state.selected.sector_models.includes(sector_model.name)
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
                                    sector_model => this.state.selected.sector_models.includes(sector_model.name)
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
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Scenario Dependencies</div>
                    <div className="card-body">
                        <ErrorBlock errors={errors.scenario_dependencies} />
                        <DependencyList
                            name="Scenario Dependency"
                            dependencies={this.state.selected.scenario_dependencies}
                            source={
                                this.props.scenarios.filter(
                                    scenario => this.state.selected.scenarios.includes(scenario.name)
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
                                    sector_model => this.state.selected.sector_models.includes(sector_model.name)
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
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Narratives</div>
                    <div className="card-body">
                        <ErrorBlock errors={errors.narratives} />
                        <NarrativeList
                            name="Narrative"
                            narratives={selected.narratives}
                            sector_models={this.props.sector_models.filter(
                                sector_model => this.state.selected.sector_models.includes(sector_model.name)
                            )}
                            onChange={(narratives) => this.handleChange('narratives', narratives)}  />
                    </div>

                </div>

                <PrimaryButton value="Save" onClick={this.handleSave} />
                <SecondaryButton value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SosModelConfigForm.propTypes = {
    sos_model: PropTypes.object.isRequired,
    model_runs: PropTypes.array.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
    save: PropTypes.bool,
    onNavigate: PropTypes.func,
    onSave: PropTypes.func,
    onCancel: PropTypes.func,
    onEdit: PropTypes.func
}

export default SosModelConfigForm
