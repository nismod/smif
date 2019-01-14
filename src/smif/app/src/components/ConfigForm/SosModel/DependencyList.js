import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import {PrimaryButton, SecondaryButton, SuccessButton, DangerButton} from 'components/ConfigForm/General/Buttons'

class DependencyList extends Component {
    constructor(props) {
        super(props)

        this.state = {
            formPopupIsOpen: false,
            formEditMode: false,
            formEditNumber: -1,
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
        let new_dep = Object.assign([], this.props.dependencies)

        // Prevent user from adding existing dependencies
        if (this.props.dependencies.filter((dependency, idx) => {
            return (
                dependency.source == this.state.dependency.source &&
                dependency.source_output == this.state.dependency.source_output &&
                dependency.sink == this.state.dependency.sink &&
                dependency.sink_input == this.state.dependency.sink_input &&
                idx != parseInt(this.state.formEditNumber)
            )
        }).length > 0) {
            alert('Cannot save dependency because it already exists')
        }
        // Add dependency
        else {
            if (this.state.formEditMode) {
                new_dep[this.state.formEditNumber] = this.state.dependency
            }
            else {
                new_dep.push(this.state.dependency)
            }
            this.props.onChange(new_dep)
            this.closeForm()
        }
    }

    emptyForm() {
        return {
            source: '',
            source_output: '',
            sink: '',
            sink_input: '',
            timestep: 'CURRENT'
        }
    }

    handleCreate() {
        this.setState({dependency: this.emptyForm()})
        this.setState({formEditNumber: -1})
        this.openForm()
    }

    handleEdit(event) {
        const target = event.currentTarget
        const name = target.dataset.name

        let dependency = Object.assign({}, this.props.dependencies[name])

        // reset invalid dependencies to force user make a new selection
        if (!this.props.source.map(source => source.name).includes(dependency.source)) {
            dependency.source = ''
            dependency.source_output = ''
        }

        if (dependency.source in this.props.source_output) {
            if (!this.props.source_output[dependency.source].map(source_output => source_output.name).includes(dependency.source_output)) {
                dependency.source_output = ''
            }
        }
        else {
            dependency.source_output = ''
        }

        if (!this.props.sink.map(sink => sink.name).includes(dependency.sink)) {
            dependency.sink = ''
            dependency.sink_input = ''
        }

        if (dependency.sink in this.props.sink_input) {
            if (!this.props.sink_input[dependency.sink].map(sink_input => sink_input.name).includes(dependency.sink_input)) {
                dependency.sink_input = ''
            }
        }
        else {
            dependency.sink_input = ''
        }

        this.setState({dependency: dependency})
        this.setState({
            formEditMode: true,
            formEditNumber: name
        })

        this.openForm()
    }

    handleDelete(name) {
        let new_dep = Object.assign([], this.props.dependencies)
        new_dep.splice(name, 1)

        this.props.onChange(new_dep)
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
                                        {
                                            this.props.source.map(source => source.name).includes(dependency.source)
                                                ? dependency.source
                                                : (<s>{dependency.source}</s>)
                                        }
                                    </td>
                                    <td className="col-text">
                                        {
                                            dependency.source in this.props.source_output
                                                ? this.props.source_output[dependency.source].map(source_output => source_output.name).includes(dependency.source_output)
                                                    ? dependency.source_output
                                                    : (<s>{dependency.source_output}</s>)
                                                : (<s>{dependency.source_output}</s>)
                                        }
                                    </td>
                                    <td className="col-text">
                                        {
                                            this.props.sink.map(sink => sink.name).includes(dependency.sink)
                                                ? dependency.sink
                                                : (<s>{dependency.sink}</s>)
                                        }
                                    </td>
                                    <td className="col-text">
                                        {
                                            dependency.sink in this.props.sink_input
                                                ? this.props.sink_input[dependency.sink].map(sink_input => sink_input.name).includes(dependency.sink_input)
                                                    ? dependency.sink_input
                                                    : (<s>{dependency.sink_input}</s>)
                                                : (<s>{dependency.sink_input}</s>)
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
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Timestep</label>
                                        <select
                                            id={name + '_timestep'}
                                            className='form-control'
                                            name="timestep"
                                            value={this.state.dependency.timestep}
                                            onChange={this.handleFormInput}>
                                            <option value='CURRENT'>Within timestep (default)</option>
                                            <option value='PREVIOUS'>Previous timestep</option>
                                        </select>
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
    sink_input: PropTypes.object.isRequired,
    onChange: PropTypes.func
}

export default DependencyList
