import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from './General/Popup.js'
import PropertySelector from './General/PropertySelector.js'
import InputsOutputsForm from './SectorModel/InputsOutputsForm.js'
import ParameterSelector from './SectorModel/ParameterSelector.js'
import PropertyList from './General/PropertyList.js'
import DeleteForm from '../../components/ConfigForm/General/DeleteForm.js'

class SectorModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selectedSectorModel: this.props.sectorModel,
            deletePopupIsOpen: false
        }

        this.closeDeletePopup = this.closeDeletePopup.bind(this)
        this.openDeletePopup = this.openDeletePopup.bind(this)
        this.handleDelete = this.handleDelete.bind(this)
    }

    componentDidMount(){
        document.addEventListener("keydown", this.handleKeyPress, false)
    }

    componentWillUnmount(){
        document.removeEventListener("keydown", this.handleKeyPress, false)
    }

    handleKeyPress(){
        if(event.keyCode === 27) {
            this.handleCancel()
        }
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedSectorModel: update(this.state.selectedSectorModel, {[name]: {$set: value}})
        })
    }

    handleDelete(config) {

        const {deletePopupType, selectedSectorModel} = this.state

        switch(deletePopupType) {
            case 'inputs':
                for (let i = 0; i < selectedSectorModel.inputs.length; i++) {
                    if (selectedSectorModel.inputs[i].name == config)
                        selectedSectorModel.inputs.splice(i, 1)
                }
                break

            case 'outputs':
                for (let i = 0; i < selectedSectorModel.outputs.length; i++) {
                    if (selectedSectorModel.outputs[i].name == config)
                        selectedSectorModel.outputs.splice(i, 1)
                }
                break

            case 'parameters':
                for (let i = 0; i < selectedSectorModel.parameters.length; i++) {
                    if (selectedSectorModel.parameters[i].name == config)
                        selectedSectorModel.parameters.splice(i, 1)
                }
                break
        }
        
        this.forceUpdate()
        this.closeDeletePopup()
    }

    handleSave() {
        this.props.saveSectorModel(this.state.selectedSectorModel)
    }

    handleCancel() {
        this.props.cancelSectorModel()
    }

    openDeletePopup(event) {

        let target_in_use_by = []

        switch(event.target.name) {
            case 'inputs' || 'parameters':
                this.props.sosModels.forEach(function(sos_model) {   
                    sos_model.dependencies.forEach(function(dependency) {
                        if (dependency.sink_model_input == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })
                break
            case 'outputs':
                this.props.sosModels.forEach(function(sos_model) {   
                    sos_model.dependencies.forEach(function(dependency) {
                        if (dependency.source_model_output == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })
                break
        }

        this.setState({
            deletePopupIsOpen: true,
            deletePopupConfigName: event.target.value,
            deletePopupType: event.target.name,
            deletePopupInUseBy: target_in_use_by
        })
    }

    closeDeletePopup() {
        this.setState({deletePopupIsOpen: false})
    }

    render() {
        const {selectedSectorModel} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="sector_model_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedSectorModel.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="sector_model_description" className="form-control" name="description" rows="5" defaultValue={selectedSectorModel.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Environment</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Class Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="classname" type="text" defaultValue={selectedSectorModel.classname} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Path</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="path" type="text" defaultValue={selectedSectorModel.path} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Inputs</div>
                        <div className="card-body">
                            <PropertyList itemsName="inputs" items={selectedSectorModel.inputs} columns={{name: 'Name', spatial_resolution: 'Spatial Resolution', temporal_resolution: 'Temporal Resolution', units: 'Units'}} editButton={false} deleteButton={true} onDelete={this.openDeletePopup} />
                            <InputsOutputsForm items={selectedSectorModel.inputs} isInputs={true} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Outputs</div>
                        <div className="card-body">
                            <PropertyList itemsName="outputs" items={selectedSectorModel.outputs} columns={{name: 'Name', spatial_resolution: 'Spatial Resolution', temporal_resolution: 'Temporal Resolution', units: 'Units'}} editButton={false} deleteButton={true} onDelete={this.openDeletePopup} />
                            <InputsOutputsForm items={selectedSectorModel.outputs} isOutputs={true} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Parameters</div>
                        <div className="card-body">
                            <PropertyList itemsName="parameters" items={selectedSectorModel.parameters} columns={{name: 'Name', description: 'Description', default_value: 'Default Value', units: 'Units', absolute_range: 'Absolute Range', suggested_range: 'Suggested Range'}} editButton={false} deleteButton={true} onDelete={this.openDeletePopup} />
                            <ParameterSelector parameters={selectedSectorModel.parameters} onChange={this.handleChange}/>
                        </div>
                    </div>

                    <br/>
                </form>

                <Popup onRequestOpen={this.state.deletePopupIsOpen}>
                    <DeleteForm config_name={this.state.deletePopupConfigName} config_type={this.state.deletePopupType} in_use_by={this.state.deletePopupInUseBy} submit={this.handleDelete} cancel={this.closeDeletePopup}/>
                </Popup>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SectorModelConfigForm.propTypes = {
    sosModels: PropTypes.array.isRequired,
    sectorModel: PropTypes.object.isRequired,
    saveSectorModel: PropTypes.func,
    cancelSectorModel: PropTypes.func
}

export default SectorModelConfigForm
