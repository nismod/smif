import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class NarrativeConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedNarrative = this.props.narrative
        
        this.handleChange = this.handleChange.bind(this)
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedNarrative: update(this.state.selectedNarrative, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveNarrative(this.state.selectedNarrative)
    }

    handleCancel() {
        this.props.cancelNarrative()
    }

    render() {
        const {narrativeSets} = this.props
        const {selectedNarrative} = this.state

        console.log(selectedNarrative)
     
        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="name" type="text" disabled="true" defaultValue={selectedNarrative.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea className="form-control" name="description" rows="5" defaultValue={selectedNarrative.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Settings</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Narrative Set</label>
                                <div className="col-sm-10">
                                     
                                    <input className="form-control" name="narrative_set" list="narrative_sets" type="text" defaultValue={selectedNarrative.narrative_set} onChange={this.handleChange}/>
                                    <datalist id="narrative_sets">
                                        {
                                            narrativeSets.map(narrativeSet =>
                                                <option key={narrativeSet.name} value={narrativeSet.name}/>
                                            )
                                        }
                                    </datalist>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Parameters</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Filename</label>
                                <div className="col-sm-10">

                                    <input className="form-control" name="filename" type="text" defaultValue={selectedNarrative.filename} onChange={this.handleChange}/>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                </form>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save Sector Model Configuration" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

NarrativeConfigForm.propTypes = {
    narrative: PropTypes.object.isRequired,
    narrativeSets: PropTypes.array.isRequired,
    saveNarrative: PropTypes.func.isRequired,
    cancelNarrative: PropTypes.func.isRequired
}

export default NarrativeConfigForm