import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import Select from 'react-select'
import {PrimaryButton, SecondaryButton, SuccessButton, DangerButton} from 'components/ConfigForm/General/Buttons'

class SpecList extends Component {
    constructor(props) {
        super(props)

        this.state = {
            formPopupIsOpen: false,
            formEditMode: false,
            spec: {}
        }

        this.handleFormInput = this.handleFormInput.bind(this)

        this.handleCreate = this.handleCreate.bind(this)
        this.handleEdit = this.handleEdit.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
        this.handleDelete = this.handleDelete.bind(this)
    }

    openForm() {
        this.setState({formPopupIsOpen: true})
    }

    closeForm() {
        this.setState({formPopupIsOpen: false})
        this.setState({formEditMode: false})
    }

    handleFormInput(event) {
        this.setState({
            spec: update(this.state.spec, {[event.target.name]: {$set: event.target.value}})
        })
    }
    
    handleSubmit(event) {
        event.preventDefault()

        let new_spec = {}

        new_spec.name = this.state.spec.name

        if (this.state.spec.description.length != '') {
            new_spec.description = this.state.spec.description
        }
        if (this.state.spec.dims != []) {
            new_spec.dims = []
            this.state.spec.dims.map((dim => {
                new_spec.dims.push(dim.value)
            }))            
        }
        if (this.state.spec.default != '') {
            new_spec.default = this.state.spec.default
        }
        if (this.state.spec.unit != '') {
            new_spec.unit = this.state.spec.unit
        }
        if (this.state.spec.dtype != '') {
            new_spec.dtype = this.state.spec.dtype
        }
        if (this.state.spec.abs_range_min != '' && this.state.spec.abs_range_max != '') {
            new_spec.abs_range = [
                parseFloat(this.state.spec.abs_range_min),
                parseFloat(this.state.spec.abs_range_max)
            ]
        }
        if (this.state.spec.sug_range_min != '' && this.state.spec.sug_range_max != '') {
            new_spec.sug_range = [
                parseFloat(this.state.spec.sug_range_min),
                parseFloat(this.state.spec.sug_range_max)
            ]
        }
 
        // update specs
        let index = this.props.specs.findIndex(spec => spec.name === new_spec.name)
        if (index >= 0) {
            this.props.specs.splice(index, 1, new_spec)
        }
        else {
            this.props.specs.push(new_spec)
        }
        this.closeForm()
        this.props.onChange({
            target: {
                name: this.props.name,
                value: this.props.specs
            }
        })
    }

    emptyForm() {
        return {
            name: '',
            description: '',
            dims: [],
            default: '',
            unit: '',
            dtype: '',
            abs_range_min: '',
            abs_range_max: '',
            sug_range_min: '',
            sug_range_max: ''
        }
    }

    handleCreate() {
        this.setState({spec: this.emptyForm()})
        this.openForm()
    }

    handleEdit(event) {
        const target = event.currentTarget
        const name = target.dataset.name

        let selectedSpec = Object.assign({}, this.props.specs.filter(spec => spec.name == name)[0])

        if (selectedSpec.dims != undefined) {
            selectedSpec.dims = selectedSpec.dims.map((dim) => ({
                value: dim,
                label: dim
            }))
        }

        if (selectedSpec.default != undefined) {
            selectedSpec.default = selectedSpec.default.toString()
        }

        if (selectedSpec.abs_range != undefined) {
            selectedSpec.abs_range_min = selectedSpec.abs_range[0].toString()
            selectedSpec.abs_range_max = selectedSpec.abs_range[1].toString()
            delete selectedSpec.abs_range
        }
        if (selectedSpec.sug_range != undefined) {
            selectedSpec.sug_range_min = selectedSpec.sug_range[0].toString()
            selectedSpec.sug_range_max = selectedSpec.sug_range[1].toString()
            delete selectedSpec.sug_range
        }

        this.setState({spec: {
            ...this.emptyForm(),
            ...selectedSpec}
        })
        this.setState({formEditMode: true})
        this.openForm()
    }

    handleDelete(name) {
        this.props.specs.splice(this.props.specs.findIndex(spec => spec.name === name), 1)
        this.closeForm()
        this.props.onChange({
            target: {
                name: this.props.name,
                value: this.props.specs
            }
        })
    }

    renderSpecList(name, specs) {
        var columns = []
        columns.push('Name')
        columns.push('Dimensions')
        if (this.props.enable_defaults) {
            columns.push('Default')
        }
        columns.push('Unit')
        columns.push('DType')
        columns.push('Range')
        
        return (
            <div>
                <table className="table table-hover table-list">
                    <thead className="thead-light">
                        <tr>
                            {
                                columns.map((column) => (
                                    <th className="col-text"
                                        scope="col" key={name + '_column_' + column}>
                                        {column}
                                    </th> 
                                ))
                            }
                        </tr>
                    </thead>
                    <tbody>
                        {
                            specs.map((spec) => (
                                <tr key={spec.name}
                                    data-name={spec.name}
                                    onClick={(e) => this.handleEdit(e)}>
                                    <td className="col-text">
                                        <div title={spec.description}>
                                            {spec.name}
                                        </div>
                                    </td>
                                    <td className="col-text">
                                        {
                                            spec.dims != undefined ?
                                                spec.dims.map((dim, idx) => 
                                                    idx == 0 ? dim : ', ' + dim
                                                ) : null
                                        }
                                    </td>
                                    <td className="col-text" hidden={!this.props.enable_defaults}>
                                        {spec.default}
                                    </td>
                                    <td className="col-text">
                                        {spec.unit}
                                    </td>
                                    <td className="col-text">
                                        {spec.dtype}
                                    </td>
                                    <td className="col-text">
                                        {
                                            spec.abs_range != undefined ?
                                                'abs(' + spec.abs_range[0] + ', ' + spec.abs_range[1] + ')' : null
                                        }
                                        {
                                            spec.sug_range != undefined ?
                                                ' sug(' + spec.sug_range[0] + ', ' + spec.sug_range[1] + ')' : null
                                        }
                                    </td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>

                <SuccessButton id={'btn_add_' + name} value={'Add ' + name} onClick={() => this.handleCreate()} />
                <Popup name={'popup_add_' + name} onRequestOpen={this.state.formPopupIsOpen}>
                    <form className="form-config" onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit(e)}}>
                        <div>
                            <div className="container">
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Name</label>
                                        <input 
                                            id={name + '_spec_name'}
                                            className='form-control'
                                            type="text" 
                                            name='name'
                                            disabled={this.state.formEditMode}
                                            value={this.state.spec.name} 
                                            onChange={this.handleFormInput}
                                            autoFocus 
                                            required
                                        />
                                    </div>
                                    <div className="col">
                                        <label>Description</label>
                                        <input 
                                            id={name + '_spec_description'} 
                                            className='form-control'
                                            type="text"
                                            name="description" 
                                            value={this.state.spec.description} 
                                            onChange={this.handleFormInput}
                                            required
                                        />
                                    </div>
                                </div>

                                <label>Dimensions</label>
                                <div className="row">
                                    <div className="col">
                                        <Select
                                            isMulti
                                            onChange={(e) => this.handleFormInput(
                                                {
                                                    target: {
                                                        value: e,
                                                        name: 'dims'
                                                    }
                                                }
                                            )}
                                            value={this.state.spec.dims} 
                                            options={this.props.dims}
                                        />
                                    </div> 
                                </div>

                                <div className="row">
                                    <div className="col" hidden={!this.props.enable_defaults}>
                                        <label>Default</label>
                                        <input 
                                            id={name + '_spec_default'}
                                            className='form-control'
                                            type="text"
                                            name='default' 
                                            value={this.state.spec.default} 
                                            onChange={this.handleFormInput} 
                                        />
                                    </div>
                                    <div className="col">
                                        <label>Unit</label>
                                        <input 
                                            id={name + '_spec_unit'} 
                                            className='form-control'
                                            type="text"
                                            name="unit" 
                                            value={this.state.spec.unit} 
                                            onChange={this.handleFormInput}
                                            required
                                        />
                                    </div>
                                    <div className="col">
                                        <label>DType</label>
                                        <input 
                                            id={name + '_spec_dtype'}
                                            className='form-control'
                                            type="text" 
                                            name='dtype' 
                                            value={this.state.spec.dtype} 
                                            onChange={this.handleFormInput}
                                            required
                                        />
                                    </div>
                                </div>

                                <label>Absolute Range</label>
                                <div className="row">
                                    <div className="col">
                                        <input 
                                            id="parameter_absolute_range_low" 
                                            className='form-control'
                                            type="number" 
                                            step="any"
                                            name="abs_range_min" 
                                            value={this.state.spec.abs_range_min} 
                                            onChange={this.handleFormInput}
                                            placeholder="Minimum" />
                                    </div>
                                    <div className="col">
                                        <input 
                                            id="parameter_absolute_range_high" 
                                            className='form-control'
                                            type="number" 
                                            step="any"
                                            name="abs_range_max" 
                                            value={this.state.spec.abs_range_max} 
                                            onChange={this.handleFormInput}
                                            placeholder="Maximum" />
                                    </div>
                                </div>

                                <label>Suggested Range</label>
                                <div className="row">
                                    <div className="col">
                                        <input 
                                            id="parameter_suggested_range_low" 
                                            className='form-control'
                                            type="number" 
                                            step="any"
                                            name="sug_range_min" 
                                            value={this.state.spec.sug_range_min} 
                                            onChange={this.handleFormInput}
                                            placeholder="Minimum" />
                                    </div>
                                    <div className="col">
                                        <input id="parameter_suggested_range_high" 
                                            type="number" 
                                            className='form-control'
                                            step="any"
                                            name="sug_range_max" 
                                            value={this.state.spec.sug_range_max} 
                                            onChange={this.handleFormInput}
                                            placeholder="Maximum" />
                                    </div>
                                </div>
                            </div>

                            <br/>

                            <PrimaryButton id={'btn_' + name + '_save'} value="Save" />
                            <SecondaryButton id={'btn_' + name + '_cancel'} value="Cancel" onClick={() => this.closeForm()}/>
                            {
                                !this.state.formEditMode ? null : (
                                    <DangerButton  
                                        id={'btn_' + name + '_delete'} 
                                        onClick={() => this.handleDelete(this.state.spec.name)} />
                                )
                            }
                        </div>
                    </form>
                </Popup>
            </div>
        )
    }

    render() {
        const {name, specs} = this.props

        return this.renderSpecList(name, specs)
    }
}

SpecList.propTypes = {
    name: PropTypes.string.isRequired,
    onChange: PropTypes.func,
    specs: PropTypes.array.isRequired,
    dims: PropTypes.array.isRequired,
    enable_defaults: PropTypes.bool
}

SpecList.defaultProps = {
    enable_defaults: true
}

export default SpecList
