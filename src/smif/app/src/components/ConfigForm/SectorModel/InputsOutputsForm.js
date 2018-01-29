import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from '../General/Popup.js'

class InputsOutputsForm extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreateDependencypopupIsOpen: false
        }

        this.state.inputs = {
            name: '',
            units: '',
            spatial_resolution: '',
            temporal_resolution: ''
        }

        this.state.className = {
            name: 'form-control',
            units:  'form-control',
            spatial_resolution: 'form-control',
            temporal_resolution:  'form-control'
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
        const {onChange, items, isInputs, isOutputs} = this.props
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
            let newItems = items

            newItems.push(
                inputs
            )

            if (isInputs) {
                onChange(
                    {
                        target: {
                            name: 'inputs',
                            value: newItems,
                            type: 'array'
                        }
                    }
                )
            } else if (isOutputs) {
                onChange(
                    {
                        target: {
                            name: 'outputs',
                            value: newItems,
                            type: 'array'
                        }
                    }
                )
            }

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

    renderInputForm(type) {

        return (
            <div>
                <input className="btn btn-secondary btn-lg btn-block" type="button" value={'Add ' + type} onClick={this.openCreateDependencyPopup} />

                <Popup onRequestOpen={this.state.CreateDependencypopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit()}}>
                        <h2 ref={subtitle => this.subtitle = subtitle}>{'Add a new ' + type}</h2>
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <label>Name</label>
                                    <input autoFocus ref="" type="text" className={this.state.className.name} name="name" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <label>Units</label>
                                    <input ref="" type="text" className={this.state.className.units} name="units" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <label>Spatial Resolution</label>
                                    <input ref="" type="text" className={this.state.className.spatial_resolution} name="spatial_resolution" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>
                                </div>
                                <div className="col">
                                    <label>Temporal Resolution</label>
                                    <input type="text" className={this.state.className.temporal_resolution} name="temporal_resolution" onChange={this.handleChange}/>
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
            <div id="inputs_outputs_form_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="inputs_outputs_form_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="inputs_outputs_form_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {items, isInputs, isOutputs} = this.props

        if (items == undefined || items == null)
        {
            if (isInputs) {
                return this.renderDanger('Inputs are undefined')
            } else if (isOutputs) {
                return this.renderDanger('Outputs are undefined')
            }
        } else if ((isInputs && isOutputs) || (!isInputs && !isOutputs)) {
            return this.renderDanger('Item type is not selected')
        } else if (isInputs) {
            return this.renderInputForm('input')
        } else if (isOutputs) {
            return this.renderInputForm('output')
        }
    }
}

InputsOutputsForm.propTypes = {
    items: PropTypes.array,
    isInputs: PropTypes.bool,
    isOutputs: PropTypes.bool,
    onChange: PropTypes.func
}

export default InputsOutputsForm
