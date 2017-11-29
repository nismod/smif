import React, { Component } from 'react'
import PropTypes from 'prop-types'

import Popup from '../Popup.js'

import FaTrash from 'react-icons/lib/fa/trash'

class DependencySelector extends Component {
    constructor(props) {
        super(props)

        this.state = {
            CreateDependencypopupIsOpen: false
        }
        this.closeCreateDependencyPopup = this.closeCreateDependencyPopup.bind(this)
        this.openCreateDependencyPopup = this.openCreateDependencyPopup.bind(this)

        this.handleChange = this.handleChange.bind(this)
        this.handleAddDependency = this.handleAddDependency.bind(this)
    }

    handleChange(event) {
        this.setState({[event.target.name]: event.target.value})
    }

    handleAddDependency() {
        const {onChange} = this.props

        console.log(target.name)
        console.log(target.checked)

        //onChange(target.name, narrative_name, target.checked)
    }

    openCreateDependencyPopup() {
        this.setState({CreateDependencypopupIsOpen: true})
    }
    
    closeCreateDependencyPopup() {
        this.setState({CreateDependencypopupIsOpen: false})
    }

    renderDependencySelector(dependencies, sectorModels) {
        
        console.log(this.state)

        return (    
            <div>
                <table className="table table-sm fixed">
                    <thead className="thead-light">
                    
                        <tr>
                            <th width="23%" scope="col">Source Model</th>
                            <th width="23%" scope="col">Output</th>
                            <th width="23%" scope="col">Sink Model</th>
                            <th width="23%" scope="col">Input</th>
                            <th width="8%" scope="col"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            dependencies.map((dependency, i) => (
                                <tr key={i}>
                                    <td>{dependency.source_model}</td>
                                    <td>{dependency.source_model_output}</td>
                                    <td>{dependency.sink_model}</td>
                                    <td>{dependency.sink_model_input}</td>
                                    <td>
                                        <button name={i} onClick={this.onDeleteHandler}>
                                            <FaTrash/>
                                        </button>
                                    </td> 
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Add Dependency" onClick={this.openCreateDependencyPopup} />

                <Popup onRequestOpen={this.state.CreateDependencypopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleChange()}}>
                        <h2 ref={subtitle => this.subtitle = subtitle}>Add a new Dependency</h2>
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <div className="form-group">
                                        <label>Source</label>
                                        <select className="form-control" name="source_model" onChange={this.handleChange} required>
                                            {
                                                sectorModels.map((sectorModel, i) => (
                                                    <option key={i} value={sectorModel.name}>{sectorModel.name}</option>
                                                ))
                                            }
                                        </select>
                                        <input type="text" className="form-control" name="source_output" placeholder="Source Output" onChange={this.handleChange} required/>
                                    </div>
                                </div>
                                <div className="col">
                                    <div className="form-group">
                                        <label>Sink</label>
                                        <select className="form-control" name="sink_model" onChange={this.handleChange} required>
                                            {
                                                sectorModels.map((sectorModel, i) => (
                                                    <option key={i} value={sectorModel.name}>{sectorModel.name}</option>
                                                ))
                                            }
                                        </select>
                                        <input type="text" className="form-control" name="sink_input" placeholder="Sink Input" onChange={this.handleChange} required/>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </form>
                    <input className="btn btn-secondary btn-lg btn-block" type="submit" value="Add" onClick={this.handleAddDependency}/>
                    <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeCreateDependencyPopup}/>
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
        const {sosModel, sectorModels} = this.props
        
        if (sosModel == null) {
            return this.renderWarning('There is no sosModel selected')
        } else if (sosModel.dependencies == undefined) {
            return this.renderWarning('Dependencies are undefined')
        } else if (sectorModels == null) {
            return this.renderWarning('There are no sectorModels configured')
        } else {           
            return this.renderDependencySelector(sosModel.dependencies, sectorModels)
        }        
    }
}

DependencySelector.propTypes = {
    sosModel: PropTypes.object,
    sectorModels: PropTypes.array,
    onChange: PropTypes.func
}

export default DependencySelector