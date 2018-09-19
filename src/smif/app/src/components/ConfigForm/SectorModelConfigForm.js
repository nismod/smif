import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import ParameterSelector from 'components/ConfigForm/SectorModel/ParameterSelector.js'
import SpecList from 'components/ConfigForm/General/SpecList.js'
import DeleteForm from 'components/ConfigForm/General/DeleteForm.js'
import { SaveButton, CancelButton } from 'components/ConfigForm/General/Buttons'

class SectorModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selectedSectorModel: this.props.sectorModel,
            deletePopupIsOpen: false
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

    handleSave() {
        this.props.saveSectorModel(this.state.selectedSectorModel)
    }

    handleCancel() {
        this.props.cancelSectorModel()
    }

    renderSectorModelConfigForm() {
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

                    <div className="card">
                        <div className="card-header">Environment</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Class Name</label>
                                <div className="col-sm-10">
                                    <input id="sector_model_classname" className="form-control" name="classname" type="text" defaultValue={selectedSectorModel.classname} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Path</label>
                                <div className="col-sm-10">
                                    <input id="sector_model_path" className="form-control" name="path" type="text" defaultValue={selectedSectorModel.path} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">Inputs</div>
                        <div className="card-body">
                            <SpecList name="input" specs={selectedSectorModel.inputs} dims={[{value: 'country', label: 'country'}, {value: 'lad', label: 'lad'}]} />
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">Outputs</div>
                        <div className="card-body">
                            <SpecList name="output" specs={selectedSectorModel.outputs} dims={[{value: 'country', label: 'country'}, {value: 'lad', label: 'lad'}]} />
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">Parameters</div>
                        <div className="card-body">
                            <SpecList name="parameter" specs={selectedSectorModel.parameters} dims={[{value: 'country', label: 'country'}, {value: 'lad', label: 'lad'}]} />
                        </div>
                    </div>
                </form>

                <SaveButton onClick={this.handleSave} />
                <CancelButton onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div>
                <div id="alert-danger" className="alert alert-danger">
                    {message}
                </div>
                <CancelButton onClick={this.handleCancel} />
            </div>
        )
    }

    render() {
        const {selectedSectorModel} = this.state

        if (selectedSectorModel.name == undefined) {
            return this.renderDanger('This Scenario Set does not exist.')
        } else {
            return this.renderSectorModelConfigForm()
        }
    }
}

SectorModelConfigForm.propTypes = {
    sosModels: PropTypes.array.isRequired,
    sectorModel: PropTypes.object.isRequired,
    saveSectorModel: PropTypes.func,
    cancelSectorModel: PropTypes.func
}

SectorModelConfigForm.defaultProps = {
    sosModels: [],
    sectorModel: {},
}

export default SectorModelConfigForm
