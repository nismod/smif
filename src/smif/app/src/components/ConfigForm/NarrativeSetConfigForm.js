import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import { SaveButton, CancelButton } from './General/Buttons'

class NarrativeSetConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedNarrativeSet = this.props.narrativeSet
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedNarrativeSet: update(this.state.selectedNarrativeSet, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveNarrativeSet(this.state.selectedNarrativeSet)
    }

    handleCancel() {
        this.props.cancelNarrativeSet()
    }

    render() {
        const {selectedNarrativeSet} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="narrative_set_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedNarrativeSet.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="narrative_set_description" className="form-control" name="description" rows="5" defaultValue={selectedNarrativeSet.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>
                </form>

                <SaveButton id="saveNarrativeSet" onClick={this.handleSave} />
                <CancelButton id="cancelNarrativeSet" onClick={this.handleCancel} />
            </div>
        )
    }
}

NarrativeSetConfigForm.propTypes = {
    narrativeSet: PropTypes.object.isRequired,
    saveNarrativeSet: PropTypes.func,
    cancelNarrativeSet: PropTypes.func
}

export default NarrativeSetConfigForm
