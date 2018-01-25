import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from '../General/Popup.js'

class ParameterSelector extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreatePopupIsOpen: false
        }


        this.state.inputs = {
            name: '',
            description: '',
            absolute_range_min: '',
            absolute_range_max: '',
            suggested_range_min: '',
            suggested_range_max: '',
            default_value: '',
            units: ''
        }

        this.state.className = {
            name: 'form-control',
            description: 'form-control',
            absolute_range_min: 'form-control',
            absolute_range_max: 'form-control',
            suggested_range_min: 'form-control',
            suggested_range_max: 'form-control',
            default_value: 'form-control',
            units: 'form-control'
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

            let newParameter = {
                name: inputs.name,
                description: inputs.description,
                absolute_range: '(' + inputs.absolute_range_min + ', ' + inputs.absolute_range_max + ')',
                suggested_range: '(' + inputs.suggested_range_min + ', ' + inputs.suggested_range_max + ')',
                default_value: inputs.default_value,
                units: inputs.units
            }

            let newParameters = parameters
            newParameters.push(newParameter)

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

    renderParameterSelector() {

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

                                    <label>Description</label>
                                    <input type="text" className={this.state.className.description} name="description" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                    Please provide a valid input.
                                    </div>
                                </div>
                            </div>

                            <div className="row">
                                <div className="col">
                                    <label>Default Value</label>
                                </div>
                                <div className="col">
                                    <label>Units</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input type="number" className={this.state.className.default_value} name="default_value" onChange={this.handleChange}/>
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

                            <label>Absolute Range</label>
                            <div className="row">
                                <div className="col">
                                    <input type="number" className={this.state.className.absolute_range_min} name="absolute_range_min" placeholder="Minimum" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <input type="number" className={this.state.className.absolute_range_max} name="absolute_range_max" placeholder="Maximum" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                            </div>

                            <label>Suggested Range</label>
                            <div className="row">
                                <div className="col">
                                    <input type="number" className={this.state.className.suggested_range_min} name="suggested_range_min" placeholder="Minimum" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <input type="number" className={this.state.className.suggested_range_max} name="suggested_range_max" placeholder="Maximum" onChange={this.handleChange}/>
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
            <div id="parameter_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="parameter_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="parameter_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {parameters} = this.props

        if (parameters == undefined) {
            return this.renderDanger('Parameters are undefined')
        } else {
            return this.renderParameterSelector(parameters)
        }
    }
}

ParameterSelector.propTypes = {
    parameters: PropTypes.array,
    onChange: PropTypes.func,
    onDelete: PropTypes.func
}

export default ParameterSelector
