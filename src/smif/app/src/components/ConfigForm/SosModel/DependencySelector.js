import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from '../General/Popup.js'

class DependencySelector extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreateDependencypopupIsOpen: false
        }

        this.state.inputs = {
            SourceModel: '',
            SourceOutput: '',
            SinkModel: '',
            SinkInput: ''
        }

        this.state.className = {
            SourceModel: 'form-control',
            SourceOutput:  'form-control',
            SinkModel: 'form-control',
            SinkInput:  'form-control'
        }

        this.closeCreateDependencyPopup = this.closeCreateDependencyPopup.bind(this)
        this.openCreateDependencyPopup = this.openCreateDependencyPopup.bind(this)

        this.handleChange = this.handleChange.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
    }

    handleChange(event) {


        // Reset the SourceOutput and SinkInput if a difference source or sink is selected
        if (event.target.name == 'SourceModel'){
            this.setState({
                inputs: update(this.state.inputs, {
                    SourceModel: {$set: event.target.value},
                    SourceOutput: {$set: ''}
                })
            })
        } else if (event.target.name == 'SinkModel'){
            this.setState({
                inputs: update(this.state.inputs, {
                    SinkModel: {$set: event.target.value},
                    SinkInput: {$set: ''}
                })
            })
        } else {
            this.setState({
                inputs: update(this.state.inputs, {[event.target.name]: {$set: event.target.value}})
            })
        }
    }

    handleSubmit() {
        const {onChange, dependencies} = this.props
        const {inputs, className} = this.state
        const {SourceModel, SourceOutput, SinkModel, SinkInput} = this.state.inputs

        let inputOk = true

        // Check if all inputs have a value
        Object.keys(inputs).forEach(function(input) {
            if (inputs[input] == '') {
                className[input] = 'form-control is-invalid'
                inputOk = false
            } else {
                className[input] = 'form-control is-valid'
            }
        })

        // Submit change
        if (inputOk) {
            let newDependencies = dependencies

            newDependencies.push({
                source_model: SourceModel,
                source_model_output: SourceOutput,
                sink_model: SinkModel,
                sink_model_input: SinkInput
            })

            onChange(
                {
                    target: {
                        name: 'dependencies',
                        value: newDependencies,
                        type: 'array'
                    }
                }
            )

            this.closeCreateDependencyPopup()
        }

        this.forceUpdate()
    }

    openCreateDependencyPopup() {
        this.setState({CreateDependencypopupIsOpen: true})
    }

    closeCreateDependencyPopup() {

        const {inputs, className} = this.state

        this.setState({CreateDependencypopupIsOpen: false})

        // Reset form status
        Object.keys(inputs).forEach(function(input) {
            className[input] = 'form-control'
        })
        this.setState({
            inputs: {SourceModel: '', SinkModel: '', SourceOutput: '', SinkInput: ''}
        })
    }

    renderDependencySelector(sectorModels, scenarioSets, dependencies, selectedSectorModels, selectedScenarioSets) {

        const {inputs} = this.state

        let source_selector = []
        let sink_selector = []

        // Prepare the options for the dependency selector source and sink
        if (selectedSectorModels != null && selectedScenarioSets != null) {
            source_selector.push(<option key={'source_selector_info'} disabled="disabled" value="none">Please select a source</option>)
            sink_selector.push(<option key={'sink_selector_info'} disabled="disabled" value="none">Please select a sink</option>)

            // Fill sector models
            if (selectedSectorModels.length > 0) {
                source_selector.push(<option key={'source_sectormodel_info'} disabled="disabled">Sector Model</option>)
                sink_selector.push(<option key={'sink_sectormodel_info'} disabled="disabled">Sector Model</option>)
                for(let i = 0; i < selectedSectorModels.length; i++) {
                    if (inputs['SinkModel'] != selectedSectorModels[i]) {
                        source_selector.push(<option key={'source_selector_sectormodel_' + i} value={selectedSectorModels[i]}>{selectedSectorModels[i]}</option>)
                    }
                    if (inputs['SourceModel'] != selectedSectorModels[i]) {
                        sink_selector.push(<option key={'sink_selector_sectormodel_' + i} value={selectedSectorModels[i]}>{selectedSectorModels[i]}</option>)
                    }
                }
            }
            
            // If no sector models available for sink, write not available
            if (sink_selector.length <= 2) {
                sink_selector = <option key={'sink_selector_info'} disabled="disabled" value="none">No sink available</option>
            }
            
            // Fill scenario sets
            if (selectedScenarioSets.length > 0) {
                source_selector.push(<option key={'selectedScenariosets_info'} disabled="disabled">Scenario Set</option>)
                for(let i = 0; i < selectedScenarioSets.length; i++) {
                    source_selector.push(<option key={'source_selector_scenarioset_' + i} value={selectedScenarioSets[i]}>{selectedScenarioSets[i]}</option>)
                }
            }
            
            // If no sector models and scenario sets available for source, write not available
            if (source_selector.length <= 1) {
                source_selector = <option key={'source_selector_info'} disabled="disabled" value="none">No source available</option>
            }
        }
        
        // Prepare options for source output selector
        let source_output_selector = []
        if (inputs.SourceModel != '') {
            source_output_selector.push(<option key={'source_output_selector_info'} disabled="disabled" value="none">Please select a source output</option>)
            
            let sectormodel_source_outputs = sectorModels.filter(sectorModel => sectorModel.name == inputs.SourceModel)
            let scenarioset_source_outputs = scenarioSets.filter(scenarioSet => scenarioSet.name == inputs.SourceModel)
            
            if (sectormodel_source_outputs.length == 1 && scenarioset_source_outputs.length == 0) {
                sectormodel_source_outputs[0].outputs.map(output => 
                    source_output_selector.push(
                        <option key={'source_output_' + output['name']} value={output['name']}>{output['name']}</option>
                    )
                )

                // set state for default selection
                if (scenarioset_source_outputs[0].facets.length > 0) {
                    this.state.inputs.SourceOutput = sectormodel_source_outputs[0].outputs[0].name
                } else {
                    this.state.inputs.SourceOutput = ''
                }

            } else if (scenarioset_source_outputs.length == 1 && sectormodel_source_outputs.length == 0) {
                scenarioset_source_outputs[0].facets.map(facet => 
                    source_output_selector.push(
                        <option key={'source_output_' + facet['name']} value={facet['name']}>{facet['name']}</option>
                    )
                )
            } else if ((sectormodel_source_outputs.length + scenarioset_source_outputs.length) > 1) {
                source_output_selector.push(<option key={'source_output_selector_info'} disabled="disabled" value="none">Error: Duplicates</option>)
            } else {
                source_output_selector.push(<option key={'source_output_selector_info'} disabled="disabled" value="none">None</option>)
            }

        } else {
            source_output_selector.push(<option key={'source_output_selector_info'} disabled="disabled" value="none">None</option>)
        }

        // Prepare options for sink input selector
        let sink_input_selector = []
        
        if (inputs.SinkModel != '') {
            sink_input_selector.push(<option key={'sink_input_selector_info'} disabled="disabled" value="none">Please select a sink input</option>)
            
            let sectormodel_sink_inputs = sectorModels.filter(sectorModel => sectorModel.name == inputs.SinkModel)
            sectormodel_sink_inputs[0].inputs.map(input => 
                sink_input_selector.push(
                    <option key={'sink_input_' + input['name']} value={input['name']}>{input['name']}</option>
                )
            )
        } else {
            sink_input_selector.push(<option key={'sink_input_selector_info'} disabled="disabled" value="none">None</option>)
        }
        
        return (
            <div>
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Add Dependency" onClick={this.openCreateDependencyPopup} />
                <Popup onRequestOpen={this.state.CreateDependencypopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit()}}>
                        <h2 ref={subtitle => this.subtitle = subtitle}>Add a new Dependency</h2>
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <label>Source</label>
                                    <select autoFocus className={this.state.className.SourceModel} name="SourceModel" value={this.state.inputs.SourceModel} onChange={this.handleChange}>
                                        {source_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <label>Sink</label>
                                    <select className={this.state.className.SinkModel} name="SinkModel" value={this.state.inputs.SinkModel} onChange={this.handleChange}>
                                        {sink_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <select className={this.state.className.SourceOutput} name="SourceOutput" value={this.state.inputs.SourceOutput} onChange={this.handleChange}>
                                        {source_output_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <select className={this.state.className.SinkInput} name="SinkInput" value={this.state.inputs.SinkInput} onChange={this.handleChange}>
                                        {sink_input_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                            </div>
                        </div>

                        <br/>

                        <input className="btn btn-secondary btn-lg btn-block" type="submit" value="Add"/>
                        <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeCreateDependencyPopup}/>
                    </form>

                </Popup>

            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="dependency_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="dependency_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="dependency_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {sectorModels, scenarioSets, dependencies, selectedSectorModels, selectedScenarioSets} = this.props

        if (sectorModels == null || sectorModels == undefined) {
            return this.renderDanger('sectorModels are undefined')
        } else if (scenarioSets == null || scenarioSets == undefined) {
            return this.renderDanger('scenarioSets are undefined')
        } else if (dependencies == null || dependencies == undefined) {
            return this.renderDanger('Dependencies are undefined')
        } else if (selectedSectorModels == null || selectedSectorModels == undefined) {
            return this.renderDanger('selectedSectorModels are undefined')
        } else if (selectedScenarioSets == null || selectedScenarioSets == undefined) {
            return this.renderDanger('selectedScenarioSets are undefined')
        } else if ((selectedSectorModels == null || selectedSectorModels == undefined) || (selectedScenarioSets == null || selectedScenarioSets == undefined)) {
            if (selectedSectorModels == null || selectedSectorModels == undefined) {
                return this.renderDependencySelector(sectorModels, scenarioSets, dependencies, [], selectedScenarioSets)
            } else if (selectedScenarioSets == null || selectedScenarioSets == undefined) {
                return this.renderDependencySelector(sectorModels, scenarioSets, dependencies, selectedSectorModels, [])
            }
        } else {
            return this.renderDependencySelector(sectorModels, scenarioSets, dependencies, selectedSectorModels, selectedScenarioSets)
        }
    }
}

DependencySelector.propTypes = {
    sectorModels: PropTypes.array,
    scenarioSets: PropTypes.array,
    dependencies: PropTypes.array,
    selectedSectorModels: PropTypes.array,
    selectedScenarioSets: PropTypes.array,
    onChange: PropTypes.func,
    onDelete: PropTypes.func
}

export default DependencySelector
