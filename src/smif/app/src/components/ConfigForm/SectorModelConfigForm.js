import React, {Component} from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import SpecList from 'components/ConfigForm/General/SpecList.js'
import {PrimaryButton, SecondaryButton} from 'components/ConfigForm/General/Buttons'

class SectorModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selected: this.props.sector_model,
            form_inUse_Dropdown_isOpen: false
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

        let dims = this.props.dimensions.map(dim => ({
            value: dim.name,
            label: dim.name
        }))

        dims.sort(function(a, b){
            if(a.label < b.label) { return -1 }
            if(a.label > b.label) { return 1 }
            return 0
        })

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input 
                                    className="form-control" 
                                    type="text" 
                                    disabled="true" 
                                    defaultValue={selected.name} 
                                    onChange={(event) => this.handleChange('name', event.target.value)}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea 
                                    className="form-control" 
                                    name="description" 
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
                                        System-of-Systems Model Configuration
                                    </button>
                                    <div className={`dropdown-menu${this.state.inuse_dropdown ? ' show' : ''}`} aria-labelledby="dropdownMenuButton">
                                        {
                                            this.props.sos_models.filter(sos_model => sos_model.sector_models.includes(this.props.sector_model.name)).map(sector_model => (
                                                <a key={sector_model.name} 
                                                    className="btn dropdown-item" 
                                                    onClick={() => this.props.onNavigate('/configure/sos-models/' + sector_model.name)}>
                                                    {sector_model.name}
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
                    <div className="card-header">Environment</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Class Name</label>
                            <div className="col-sm-10">
                                <input 
                                    className="form-control" 
                                    type="text" 
                                    defaultValue={selected.classname} 
                                    onChange={(event) => this.handleChange('classname', event.target.value)}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Path</label>
                            <div className="col-sm-10">
                                <input 
                                    className="form-control" 
                                    type="text" 
                                    defaultValue={selected.path} 
                                    onChange={(event) => this.handleChange('path', event.target.value)}/>
                            </div>
                        </div>

                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Inputs</div>
                    <div className="card-body">
                        <SpecList 
                            name="inputs" 
                            specs={selected.inputs} 
                            dims={dims}
                            enable_defaults={false}
                            onChange={(event) => this.handleChange('inputs', event.target.value)}/>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Outputs</div>
                    <div className="card-body">
                        <SpecList 
                            name="outputs" 
                            specs={selected.outputs} 
                            dims={dims}
                            enable_defaults={false}
                            onChange={(event) => this.handleChange('outputs', event.target.value)}/>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Parameters</div>
                    <div className="card-body">
                        <SpecList 
                            name="parameters" 
                            specs={selected.parameters} 
                            dims={dims}
                            onChange={(event) => this.handleChange('parameters', event.target.value)}/>
                    </div>
                </div>

                <PrimaryButton value="Save" onClick={this.handleSave} />
                <SecondaryButton value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

SectorModelConfigForm.propTypes = {
    sos_models: PropTypes.array.isRequired,
    sector_model: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    saveSectorModel: PropTypes.func,
    cancelSectorModel: PropTypes.func,
    save: PropTypes.bool,
    onNavigate: PropTypes.func,
    onSave: PropTypes.func,
    onCancel: PropTypes.func,
    onEdit: PropTypes.func
}

SectorModelConfigForm.defaultProps = {
    sos_models: [],
    sector_model: {},
}

export default SectorModelConfigForm
