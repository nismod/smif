import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from '../General/Popup.js'

class ParameterFileSelector extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreatePopupIsOpen: false
        }

        this.state.inputs = {
            name: '',
            filename: '',
            spatial_resolution: '',
            temporal_resolution: '',
            units: '',
        }

        this.state.className = {
            name: 'form-control',
            filename: 'form-control',
            spatial_resolution: 'form-control',
            temporal_resolution: 'form-control',
            units: 'form-control',
        }

        this.closeCreatePopup = this.closeCreatePopup.bind(this)
        this.openCreatePopup = this.openCreatePopup.bind(this)

        this.handleChange = this.handleChange.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
    }

    handleChange(event) {

        this.setState({
            inputs: update(this.state.inputs, {[event.target.name]: {$set: event.target.value}})
        })
    }

    handleSubmit() {
        const {onChange, parameters} = this.props
        const {inputs, className} = this.state

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

            let newParameters = parameters
            newParameters.push(inputs)

            onChange(
                {
                    target: {
                        name: 'parameters',
                        value: newParameters,
                        type: 'array'
                    }
                }
            )

            this.closeCreatePopup()
        }

        this.forceUpdate()
    }

    openCreatePopup() {
        this.setState({CreatePopupIsOpen: true})
    }

    closeCreatePopup() {

        const {inputs, className} = this.state

        this.setState({CreatePopupIsOpen: false})

        // Reset form status
        Object.keys(inputs).forEach(function(input) {
            className[input] = 'form-control'
        })
    }

    renderParameterFileSelector() {

        return (
            <div>
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Add Parameter" onClick={this.openCreatePopup} />

                <Popup onRequestOpen={this.state.CreatePopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit()}}>
                        <h2 ref={subtitle => this.subtitle = subtitle}>Add a new Parameter</h2>
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <label>Name</label>
                                    <input autoFocus type="text" className={this.state.className.name} name="name" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                    Please provide a valid input.
                                    </div>
                                </div>
                            </div>

                            <div className="row">
                                <div className="col">
                                    <label>Filename</label>
                                </div>
                                <div className="col">
                                    <label>Units</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input type="text" className={this.state.className.filename} name="filename" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                    Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <input type="text" className={this.state.className.units} name="units" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                    Please provide a valid input.
                                    </div>
                                </div>
                            </div>

                            <div className="row">
                                <div className="col">
                                    <label>Spatial Resolution</label>
                                </div>
                                <div className="col">
                                    <label>Temporal Resolution</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input type="text" className={this.state.className.spatial_resolution} name="spatial_resolution" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <input type="text" className={this.state.className.temporal_resolution} name="temporal_resolution" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                Please provide a valid input.
                                    </div>
                                </div>
                            </div>
                        </div>

                        <br/>

                        <input className="btn btn-secondary btn-lg btn-block" type="submit" value="Add"/>
                        <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeCreatePopup}/>
                    </form>

                </Popup>

            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="parameter_file_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="parameter_file_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="parameter_file_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {parameters} = this.props

        if (parameters == undefined) {
            return this.renderDanger('Parameters are undefined')
        } else {
            return this.renderParameterFileSelector(parameters)
        }
    }
}

ParameterFileSelector.propTypes = {
    parameters: PropTypes.array,
    onChange: PropTypes.func,
    onDelete: PropTypes.func
}

export default ParameterFileSelector
