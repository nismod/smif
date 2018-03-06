import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class NarrativeConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedNarrative = this.props.narrative
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

        let selectedNarrativeSet = {name: '', description: ''}
        let narrativeSetSelected = false

        if (selectedNarrative.narrative_set != '') {
            selectedNarrativeSet = narrativeSets.filter(narrativeSet => narrativeSet.name == selectedNarrative.narrative_set)[0]
            narrativeSetSelected = true
        }

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="narrative_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedNarrative.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="narrative_description" className="form-control" name="description" rows="5" defaultValue={selectedNarrative.description} onChange={this.handleChange}/>
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
                                    <select className="form-control" name="narrative_set" defaultValue={selectedNarrative.narrative_set} onChange={this.handleChange}>
                                        <option value="" >Please select a Narrative Set</option>
                                        {
                                            narrativeSets.map(narrativeSet =>
                                                <option key={narrativeSet.name} value={narrativeSet.name}>{narrativeSet.name}</option>
                                            )
                                        }
                                    </select>
                                    <br/>

                                    <div className="alert alert-dark" hidden={!narrativeSetSelected} role="alert">
                                        {selectedNarrativeSet && selectedNarrativeSet.description}
                                    </div>
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

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSave} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

NarrativeConfigForm.propTypes = {
    narrative: PropTypes.object.isRequired,
    narrativeSets: PropTypes.array.isRequired,
    saveNarrative: PropTypes.func,
    cancelNarrative: PropTypes.func
}

export default NarrativeConfigForm
