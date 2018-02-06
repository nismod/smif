import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class ScenarioSetConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedScenarioSet = this.props.scenarioSet
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
            selectedScenarioSet: update(this.state.selectedScenarioSet, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveScenarioSet(this.state.selectedScenarioSet)
    }

    handleCancel() {
        this.props.cancelScenarioSet()
    }

    render() {
        const {selectedScenarioSet} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="scenario_set_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedScenarioSet.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="scenario_set_description" className="form-control" name="description" rows="5" defaultValue={selectedScenarioSet.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                </form>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save Scenario Set Configuration" onClick={this.handleSave} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ScenarioSetConfigForm.propTypes = {
    scenarioSet: PropTypes.object.isRequired,
    saveScenarioSet: PropTypes.func,
    cancelScenarioSet: PropTypes.func
}

export default ScenarioSetConfigForm
