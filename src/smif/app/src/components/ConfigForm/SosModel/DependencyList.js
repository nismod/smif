import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import {SaveButton, CancelButton, CreateButton, DangerButton} from 'components/ConfigForm/General/Buttons'

class DependencyList extends Component {
    constructor(props) {
        super(props)

        this.state = {
            formPopupIsOpen: false,
            formEditMode: false,
            formEditNumber: 0,
            dependency: this.emptyForm()
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
        if (event.target.name == 'source') {
            this.setState({
                dependency: update(
                    this.state.dependency, 
                    {
                        [event.target.name]: {$set: event.target.value},
                        ['source_output']: {$set: ''}
                    }
                )
            })
        }
        else if (event.target.name == 'sink') {
            this.setState({
                dependency: update(
                    this.state.dependency, 
                    {
                        [event.target.name]: {$set: event.target.value},
                        ['sink_input']: {$set: ''}
                    }
                )
            })
        }
        else {
            this.setState({
                dependency: update(
                    this.state.dependency, 
                    {[event.target.name]: {$set: event.target.value}}
                )
            })
        }
    }
    
    handleSubmit(event) {
        event.preventDefault()

        if (this.state.formEditMode) {
            this.props.dependencies[this.state.formEditNumber] = this.state.dependency
        }
        else {
            this.props.dependencies.push(this.state.dependency)
        }
        this.closeForm()
    }

    emptyForm() {
        return {
            source: '',
            source_output: '',
            sink: '',
            sink_input: ''
        }
    }

    handleCreate() {
        this.setState({dependency: this.emptyForm()})
        this.openForm()
    }

    handleEdit(event) {
        const target = event.currentTarget
        const name = target.dataset.name

        this.setState({dependency: this.props.dependencies[name]})
        this.setState({
            formEditMode: true,
            formEditNumber: name
        })

        this.openForm()
    }

    handleDelete(name) {
        this.props.dependencies.splice(name, 1)
        this.closeForm()
    }

    render() {
        const {name, dependencies, source, source_output, sink, sink_input} = this.props
        var columns = ['Source', 'Source Output', 'Sink', 'Sink Input']

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
                            dependencies.map((dependency, idx) => (
                                <tr key={idx}
                                    data-name={idx}
                                    onClick={(e) => this.handleEdit(e)}>
                                    <td className="col-text">
                                        {dependency.source}
                                    </td>
                                    <td className="col-text">
                                        {dependency.source_output}
                                    </td>
                                    <td className="col-text">
                                        {dependency.sink}
                                    </td>
                                    <td className="col-text">
                                        {dependency.sink_input}
                                    </td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>

                <CreateButton id={'btn_add_' + name} value={'Add ' + name} onClick={() => this.handleCreate()} />
                <Popup name={'popup_add_' + name} onRequestOpen={this.state.formPopupIsOpen}>
                    <form className="form-config" onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleSubmit(e)}}>
                        <div>
                            <div className="container">
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Source</label>
                                        <select 
                                            id={name + '_source'}
                                            className='form-control'
                                            name="source" 
                                            value={this.state.dependency.source}
                                            onChange={this.handleFormInput}
                                            required>
                                            <option
                                                value=''
                                                disabled>
                                                Please select 
                                            </option>
                                            {
                                                source.map((source) => (
                                                    <option 
                                                        key={source.name}
                                                        value={source.name}>
                                                        {source.name}
                                                    </option>
                                                ))
                                            }
                                        </select>
                                    </div>
                                    <div className="col">
                                        <label className='label'>Sink</label>
                                        <select 
                                            id={name + '_sink'}
                                            className='form-control'
                                            name="sink" 
                                            value={this.state.dependency.sink}
                                            onChange={this.handleFormInput}
                                            required>
                                            <option
                                                value=''
                                                disabled>
                                                Please select 
                                            </option>
                                            {
                                                sink.map((sink) => (
                                                    <option 
                                                        key={sink.name}
                                                        value={sink.name}>
                                                        {sink.name}
                                                    </option>
                                                ))
                                            }
                                        </select>
                                    </div>
                                </div>
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Source Output</label>
                                        <select 
                                            id={name + '_source_output'}
                                            className='form-control'
                                            name="source_output" 
                                            value={this.state.dependency.source_output}
                                            onChange={this.handleFormInput}
                                            required>
                                            <option
                                                value=''
                                                disabled>
                                                Please select 
                                            </option>
                                            {
                                                this.state.dependency.source != '' 
                                                    ? source_output[this.state.dependency.source].map(output => (
                                                        <option 
                                                            key={output.name}
                                                            value={output.name}>
                                                            {output.name}
                                                        </option>
                                                    )) : null
                                            }
                                        </select>
                                    </div>
                                    <div className="col">
                                        <label className='label'>Sink Input</label>
                                        <select 
                                            id={name + '_sink'}
                                            className='form-control'
                                            name="sink_input" 
                                            value={this.state.dependency.sink_input}
                                            onChange={this.handleFormInput}
                                            required>
                                            <option
                                                value=''
                                                disabled>
                                                Please select 
                                            </option>
                                            {
                                                this.state.dependency.sink != '' 
                                                    ? sink_input[this.state.dependency.sink].map(input => (
                                                        <option 
                                                            key={input.name}
                                                            value={input.name}>
                                                            {input.name}
                                                        </option>
                                                    )) : null
                                            }
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <br/>

                            <SaveButton id={'btn_' + name + '_save'}  />
                            <CancelButton id={'btn_' + name + '_cancel'} onClick={() => this.closeForm()}/>
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

DependencyList.propTypes = {
    name: PropTypes.string.isRequired,
    dependencies: PropTypes.array.isRequired,
    source: PropTypes.array.isRequired,
    source_output: PropTypes.object.isRequired,
    sink: PropTypes.array.isRequired,
    sink_input: PropTypes.object.isRequired
}

export default DependencyList
