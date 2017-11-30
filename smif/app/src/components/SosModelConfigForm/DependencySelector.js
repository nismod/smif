import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'react-addons-update'

import Popup from '../Popup.js'

import FaTrash from 'react-icons/lib/fa/trash'

class DependencySelector extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreateDependencypopupIsOpen: false
        }

        this.state.inputs = {
            SourceModelClass: 'form-control',
            SourceModel: '',
            SourceOutputClass:  'form-control',
            SourceOutput: '',
            SinkModelClass: 'form-control',
            SinkModel: '',
            SinkInputClass:  'form-control',
            SinkInput: ''
        }

        this.closeCreateDependencyPopup = this.closeCreateDependencyPopup.bind(this)
        this.openCreateDependencyPopup = this.openCreateDependencyPopup.bind(this)

        this.handleChange = this.handleChange.bind(this)
        this.handleAddDependency = this.handleAddDependency.bind(this)

    }

    handleChange(event) {
        this.setState({
            inputs: update(this.state.inputs, {[event.target.name]: {$set: event.target.value}})
        })
    }

    handleAddDependency() {
        const {onChange, dependencies} = this.props
        const {SourceModel, SourceOutput, SinkModel, SinkInput} = this.state.inputs

        // Input checking
        let inputOk = true

        if (SourceModel == '') {
            this.state.inputs.SourceModelClass = 'form-control is-invalid'
            inputOk = false
        } else {
            this.state.inputs.SourceModelClass = 'form-control is-valid'
        }

        if (SourceOutput == '') {
            this.state.inputs.SourceOutputClass = 'form-control is-invalid'
            inputOk = false
        } else {
            this.state.inputs.SourceOutputClass = 'form-control is-valid'
        }

        if (SinkModel == '') {
            this.state.inputs.SinkModelClass = 'form-control is-invalid'
            inputOk = false
        } else {
            this.state.inputs.SinkModelClass = 'form-control is-valid'
        }

        if (SinkInput == '') {
            this.state.inputs.SinkInputClass = 'form-control is-invalid'
            inputOk = false
        } else {
            this.state.inputs.SinkInputClass = 'form-control is-valid'
        }

        // Submit change if all input are ok
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
            
            this.state.inputs = {
                SourceModelClass: 'form-control',
                SourceModel: '',
                SourceOutputClass:  'form-control',
                SourceOutput: '',
                SinkModelClass: 'form-control',
                SinkModel: '',
                SinkInputClass:  'form-control',
                SinkInput: ''
            }
            this.closeCreateDependencyPopup()
        }

        this.forceUpdate()        
    }

    openCreateDependencyPopup() {
        this.setState({CreateDependencypopupIsOpen: true})
    }
    
    closeCreateDependencyPopup() {
        this.setState({CreateDependencypopupIsOpen: false})
    }

    renderDependencySelector(dependencies, sectorModels) {

        return (    
            <div>
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Add Dependency" onClick={this.openCreateDependencyPopup} />

                <Popup onRequestOpen={this.state.CreateDependencypopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleAddDependency()}}>
                        <h2 ref={subtitle => this.subtitle = subtitle}>Add a new Dependency</h2>
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <label>Source</label>
                                    <select className={this.state.inputs.SourceModelClass} name="SourceModel" defaultValue="none" onChange={this.handleChange}>
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
                                    <select className={this.state.inputs.SinkModelClass} name="SinkModel" defaultValue="none" onChange={this.handleChange}>
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
                                    <input ref="" type="text" className={this.state.inputs.SourceOutputClass} name="SourceOutput" placeholder="Source Output" onChange={this.handleChange}/>
                                    <div className="invalid-feedback">
                                            Please provide a valid input.
                                    </div>  
                                </div>
                                <div className="col">
                                    <input type="text" className={this.state.inputs.SinkInputClass} name="SinkInput" placeholder="Sink Input" onChange={this.handleChange}/>
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