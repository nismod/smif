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
        this.setState({
            inputs: update(this.state.inputs, {[event.target.name]: {$set: event.target.value}})
        })
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

    renderDependencySelector(dependencies, sectorModels, scenarioSets) {

        const {inputs} = this.state

        let source_selector = []
        let sink_selector = []

        // Prepare the options for the dependency selector source and sink
        if (sectorModels != null && scenarioSets != null) {
            source_selector.push(<option key={'source_selector_info'} disabled="disabled" value="none">Please select a source</option>)
            sink_selector.push(<option key={'sink_selector_info'} disabled="disabled" value="none">Please select a sink</option>)

            // Fill sector models
            if (sectorModels.length > 0) {
                source_selector.push(<option key={'source_sectormodel_info'} disabled="disabled">Sector Model</option>)
                sink_selector.push(<option key={'sink_sectormodel_info'} disabled="disabled">Sector Model</option>)
                for(let i = 0; i < sectorModels.length; i++) {
                    if (inputs['SinkModel'] != sectorModels[i]) {
                        source_selector.push(<option key={'source_selector_sectormodel_' + i} value={sectorModels[i]}>{sectorModels[i]}</option>)
                    }
                    if (inputs['SourceModel'] != sectorModels[i]) {
                        sink_selector.push(<option key={'sink_selector_sectormodel_' + i} value={sectorModels[i]}>{sectorModels[i]}</option>)
                    }
                }
            }
            
            // If no sector models available for sink, write not available
            if (sink_selector.length <= 2) {
                sink_selector = <option key={'sink_selector_info'} disabled="disabled" value="none">No sink available</option>
            }
            
            // Fill scenario sets
            if (scenarioSets.length > 0) {
                source_selector.push(<option key={'scenariosets_info'} disabled="disabled">Scenario Set</option>)
                for(let i = 0; i < scenarioSets.length; i++) {
                    source_selector.push(<option key={'source_selector_scenarioset_' + i} value={scenarioSets[i]}>{scenarioSets[i]}</option>)
                }
            }
            
            // If no sector models and scenario sets available for source, write not available
            if (source_selector.length <= 1) {
                source_selector = <option key={'source_selector_info'} disabled="disabled" value="none">No source available</option>
            }
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
                                    <select autoFocus className={this.state.className.SourceModel} name="SourceModel" defaultValue="none" onChange={this.handleChange}>
                                        {source_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <label>Sink</label>
                                    <select className={this.state.className.SinkModel} name="SinkModel" defaultValue="none" onChange={this.handleChange}>
                                        {sink_selector}
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input ref="" type="text" className={this.state.className.SourceOutput} name="SourceOutput" placeholder="Source Output" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <input type="text" className={this.state.className.SinkInput} name="SinkInput" placeholder="Sink Input" onChange={this.handleChange}/>
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
        const {dependencies, sectorModels, scenarioSets} = this.props

        if (dependencies == null || dependencies == undefined) {
            return this.renderDanger('Dependencies are undefined')
        } else if ((sectorModels == null || sectorModels == undefined) && (scenarioSets == null || scenarioSets == undefined)) {
            return this.renderInfo('There are no sectorModels and scenarioSets configured')
        } else if ((sectorModels == null || sectorModels == undefined) || (scenarioSets == null || scenarioSets == undefined)) {
            if (sectorModels == null || sectorModels == undefined) {
                return this.renderDependencySelector(dependencies, [], scenarioSets)
            } else if (scenarioSets == null || scenarioSets == undefined) {
                return this.renderDependencySelector(dependencies, sectorModels, [])
            }
        } else {
            return this.renderDependencySelector(dependencies, sectorModels, scenarioSets)
        }
    }
}

DependencySelector.propTypes = {
    dependencies: PropTypes.array,
    sectorModels: PropTypes.array,
    scenarioSets: PropTypes.array,
    onChange: PropTypes.func,
    onDelete: PropTypes.func
}

export default DependencySelector
