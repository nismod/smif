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
    }

    renderDependencySelector(dependencies, sectorModels) {

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
                                    <select className={this.state.className.SourceModel} name="SourceModel" defaultValue="none" onChange={this.handleChange}>
                                        <option disabled="disabled" value="none">Please select a source</option>
                                        {
                                            sectorModels.map((sectorModel, i) => (
                                                <option key={i} value={sectorModel.name}>{sectorModel.name}</option>
                                            ))
                                        }
                                    </select>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <label>Sink</label>
                                    <select className={this.state.className.SinkModel} name="SinkModel" defaultValue="none" onChange={this.handleChange}>
                                        <option disabled="disabled" value="none">Please select a sink</option>
                                        {
                                            sectorModels.map((sectorModel, i) => (
                                                <option key={i} value={sectorModel.name}>{sectorModel.name}</option>
                                            ))
                                        }
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

    renderWarning(message) {
        return (
            <div className="alert alert-danger">
                {message}
            </div>
        )
    }

    render() {
        const {dependencies, sectorModels} = this.props

        if (dependencies == undefined) {
            return this.renderWarning('Dependencies are undefined')
        } else if (sectorModels == null) {
            return this.renderWarning('There are no sectorModels configured')
        } else {
            return this.renderDependencySelector(dependencies, sectorModels)
        }
    }
}

DependencySelector.propTypes = {
    dependencies: PropTypes.array,
    sectorModels: PropTypes.array,
    onChange: PropTypes.func,
    onDelete: PropTypes.func
}

export default DependencySelector
