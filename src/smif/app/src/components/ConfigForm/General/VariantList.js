import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import {PrimaryButton, SecondaryButton, SuccessButton, DangerButton} from 'components/ConfigForm/General/Buttons'

class VariantList extends Component {
    constructor(props) {
        super(props)

        this.state = {
            formPopupIsOpen: false,
            formEditMode: false,
            formEditNumber: -1,
            variant: this.emptyForm()
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
        if (['name', 'description'].includes(event.target.name)) {
            this.setState({
                variant: update(
                    this.state.variant, 
                    {[event.target.name]: {$set: event.target.value}}
                )
            })
        } 
        else {
            let newData = Object.assign({}, this.state.variant.data)
            newData[event.target.name] = event.target.value
            
            this.setState({
                variant: update(
                    this.state.variant, 
                    {data: {$set: newData}}
                )
            })
        }
    }
    
    handleSubmit(event) {
        event.preventDefault()

        // Add provide
        if (this.state.formEditMode) {
            this.props.variants[this.state.formEditNumber] = this.state.variant
        }
        else {
            this.props.variants.push(this.state.variant)
        }

        this.closeForm()
        this.props.onChange({
            target: {
                name: this.props.name,
                value: this.props.variants
            }
        })
    }

    emptyForm() {
        return {
            name: '',
            description: '',
            data: this.props.provides.reduce(function(obj, item) {
                obj[item.name] = ''
                return obj}, {}
            )
        }
    }

    handleCreate() {
        this.setState({variant: this.emptyForm()})
        this.setState({formEditNumber: -1})
        this.openForm()
    }

    handleEdit(event) {
        const target = event.currentTarget
        const name = target.dataset.name
        
        let variant = JSON.parse(JSON.stringify(this.props.variants[name]))

        // Clean keys that not exist in provides
        Object.keys(variant.data).map(key =>
        {
            if(!this.props.provides.map(provide => provide.name).includes(key)) {
                delete variant.data[key]
            }
        })

        // Create keys for all specs that it provides but not defined
        this.props.provides.map(provide => {
            if (variant.data[provide.name] == undefined) {
                variant.data[provide.name] = ''
            }}
        )

        this.setState({variant: variant})
        this.setState({
            formEditMode: true,
            formEditNumber: name
        })

        this.openForm()
    }

    handleDelete(name) {
        this.props.variants.splice(name, 1)
        
        this.closeForm()
        this.props.onChange({
            target: {
                name: this.props.name,
                value: this.props.variants
            }
        })
    }

    render() {
        const { variants } = this.props
        var columns = ['Name', 'Description', 'Data' ]

        // Sync variants with provides
        this.props.variants.forEach((variant, idx) => {

            // Remove old keys
            Object.keys(variant.data).forEach((key) => {
                if (!this.props.provides.map(provide => provide.name).includes(key)) {
                    delete this.props.variants[idx].data[key]
                }
            })

            // Add non-defined keys
            this.props.provides.forEach(provide => {
                if (!Object.keys(variant.data).includes(provide.name)) {
                    this.props.variants[idx].data[provide.name] = ''
                }
            })

        })

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
                            variants.map((variant, idx) => (
                                <tr key={idx}
                                    data-name={idx}
                                    onClick={(e) => this.handleEdit(e)}>
                                    <td className="col-text">
                                        {variant.name}
                                    </td>
                                    <td className="col-text">
                                        {variant.description}
                                    </td>
                                    <td className="col-text">
                                        {
                                            Object.keys(variant.data).map(key => 
                                                <div key={key}>
                                                    {key + ': '}
                                                    {variant.data[key]}
                                                </div>
                                            )
                                        }
                                    </td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>

                <SuccessButton id={'btn_add_variant'} value={'Add variant'} onClick={() => this.handleCreate()} />
                <Popup name={'popup_add_' + name} onRequestOpen={this.state.formPopupIsOpen}>
                    <form className="form-config" onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit(e)}}>
                        <div>
                            <div className="container">
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Name</label>
                                        <input 
                                            id={name + '_variant_name'}
                                            className='form-control'
                                            type="text" 
                                            name='name'
                                            disabled={this.state.formEditMode}
                                            value={this.state.variant.name} 
                                            onChange={this.handleFormInput}
                                            autoFocus 
                                            required
                                        />
                                    </div>
                                    <div className="col">
                                        <label>Description</label>
                                        <input 
                                            id={name + '_variant_description'} 
                                            className='form-control'
                                            type="text"
                                            name="description" 
                                            value={this.state.variant.description} 
                                            onChange={this.handleFormInput}
                                            required
                                        />
                                    </div>
                                </div>
                                <label className='label'>Data</label>
                                {
                                    this.props.provides.map(provide =>
                                        <div key={provide.name} className="row">
                                            <div className="col">
                                                <div className="input-group-prepend">
                                                    <span className="input-group-text">{provide.name}</span>
                                                    <input
                                                        id={name + '_variant_data_'} 
                                                        className='form-control'
                                                        type="text"
                                                        name={provide.name} 
                                                        value={this.state.variant.data[provide.name]} 
                                                        onChange={this.handleFormInput}
                                                        required={this.props.require_provide_full_config}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }
                            </div>

                            <br/>

                            <PrimaryButton id={'btn_' + name + '_save'} value="Save" />
                            <SecondaryButton id={'btn_' + name + '_cancel'} value="Cancel" onClick={() => this.closeForm()}/>
                            {
                                !this.state.formEditMode ? null : (
                                    <DangerButton  
                                        id={'btn_' + name + '_delete'} 
                                        onClick={() => this.handleDelete(this.state.formEditNumber)} />
                                )
                            }
                        </div>
                    </form>
                </Popup>
            </div>
        )
    }
}

VariantList.propTypes = {
    name: PropTypes.string.isRequired,
    variants: PropTypes.array.isRequired,
    provides: PropTypes.array.isRequired,
    require_provide_full_config: PropTypes.bool,
    onChange: PropTypes.func
}

VariantList.defaultProps = {
    require_provide_full_config: false
}

export default VariantList
