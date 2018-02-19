import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from './General/Popup.js'
import PropertyList from './General/PropertyList.js'
import ScenarioConfigForm from './ScenarioSet/ScenarioConfigForm.js'
import FacetConfigForm from './ScenarioSet/FacetConfigForm.js'

class ScenarioSetConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this) 
        this.handleFacetSave = this.handleFacetSave.bind(this)
        this.handleScenarioSave = this.handleScenarioSave.bind(this)
        this.handleScenarioCreate = this.handleScenarioCreate.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.openAddFacetPopup = this.openAddFacetPopup.bind(this)
        this.openEditFacetPopup = this.openEditFacetPopup.bind(this)
        this.closeFacetPopup = this.closeFacetPopup.bind(this)

        this.openAddScenarioPopup = this.openAddScenarioPopup.bind(this)
        this.openEditScenarioPopup = this.openEditScenarioPopup.bind(this)
        this.closeScenarioPopup = this.closeScenarioPopup.bind(this)

        this.state = {}
        this.state.scenarios = this.props.scenarios
        this.state.selectedFacet = {}
        this.state.selectedScenario = {}
        this.state.selectedScenarioSet = this.props.scenarioSet
        this.state.selectedScenarios = this.props.scenarios.filter(scenario => scenario.scenario_set == this.props.scenarioSet.name)
        this.state.addFacetPopupIsOpen = false
        this.state.editScenarioPopupIsOpen = false
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

    handleFacetSave(facet) {
        const {facets} = this.state.selectedScenarioSet

        if (facets.filter(curr_facet => curr_facet.name == facet.name).length > 0) {
            for (let i = 0; i < facets.length; i++) {
                if (facets[i].name == facet.name) facets[i] = facet
            }
        } else {
            facets.push(facet)
        }
        this.forceUpdate()
        this.closeFacetPopup()
    }

    handleScenarioSave(scenario) {
        this.props.saveScenario(scenario)
        this.closeScenarioPopup()
    }

    handleScenarioCreate(scenario) {
        console.log(scenario)
        this.props.createScenario(scenario)
        this.closeScenarioPopup()
    }

    handleCancel() {
        this.props.cancelScenarioSet()
    }

    openAddFacetPopup() {
        this.setState({selectedFacet: {}})
        this.setState({addFacetPopupIsOpen: true})
    }

    closeFacetPopup() {
        this.setState({addFacetPopupIsOpen: false})
    }

    openEditFacetPopup(id) {
        this.setState({selectedFacet: this.state.selectedScenarioSet.facets[id]})
        this.setState({addFacetPopupIsOpen: true})
    }

    openAddScenarioPopup() {
        const { selectedScenarioSet} = this.state

        // prepare facets
        let new_facets = []

        for (let set_facet of selectedScenarioSet.facets) {
            new_facets.push({
                'name': set_facet.name,
                'filename': '',
                'spatial_resolution': '',
                'temporal_resolution': '',
                'units': ''
            })
        }

        this.setState({selectedScenario: { 'facets': new_facets }})
        this.setState({editScenarioPopupIsOpen: true})
    }

    closeScenarioPopup() {
        this.setState({editScenarioPopupIsOpen: false})
    }
    
    openEditScenarioPopup(id) {

        const { selectedScenarios, selectedScenarioSet} = this.state

        // update facets
        let new_facets = []

        for (let set_facet of selectedScenarioSet.facets) {
            if(selectedScenarios[id].facets.filter(scenario_facet => scenario_facet.name == set_facet.name).length) {
                // copy existing settings
                for (let i = 0; i < selectedScenarios[id].facets.length; i++) {
                    if (selectedScenarios[id].facets[i].name == set_facet.name) new_facets.push(selectedScenarios[id].facets[i])
                }
            } else {
                new_facets.push({
                    'name': set_facet.name,
                    'filename': '',
                    'spatial_resolution': '',
                    'temporal_resolution': '',
                    'units': ''
                })
            }
        }
        this.state.selectedScenarios[id].facets = new_facets

        // load scenario
        this.setState({selectedScenario: this.state.selectedScenarios[id]})
        this.setState({editScenarioPopupIsOpen: true})
    }

    render() {
        const {selectedScenarioSet, selectedScenarios, selectedScenario, selectedFacet} = this.state

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-lathis.closeScenarioPopup()bel">Name</label>
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

                <div className="card">
                    <div className="card-header">Facets</div>
                    <div className="card-body">
                        <PropertyList itemsName="facets" items={selectedScenarioSet.facets} columns={{name: 'Name', description: 'Description'}} editButton={true} deleteButton={true} onEdit={this.openEditFacetPopup} onDelete={this.handleChange} />
                        <input className="btn btn-secondary btn-lg btn-block" name="createFacet" type="button" value="Add a Facet" onClick={this.openAddFacetPopup}/>
                    </div>
                </div>

                <br/>

                <div className="card">
                    <div className="card-header">Scenarios</div>
                    <div className="card-body">
                        <PropertyList itemsName="Scenario" items={selectedScenarios} columns={{name: 'Name', description: 'Description'}} editButton={true} deleteButton={true} onEdit={this.openEditScenarioPopup} onDelete={this.handleChange} />
                        <input className="btn btn-secondary btn-lg btn-block" name="createScenario" type="button" value="Add a new Scenario" onClick={this.openAddScenarioPopup}/>
                    </div>
                </div>

                <br/>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save Scenario Set Configuration" onClick={this.handleSave} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>

                <Popup onRequestOpen={this.state.addFacetPopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation()}}>
                        <FacetConfigForm facet={selectedFacet} saveFacet={this.handleFacetSave} cancelFacet={this.closeFacetPopup}/>
                    </form>
                </Popup>

                <Popup onRequestOpen={this.state.editScenarioPopupIsOpen}>
                    <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation()}}>                
                        <ScenarioConfigForm scenario={selectedScenario} scenarioSet={selectedScenarioSet} createScenario={this.handleScenarioCreate} saveScenario={this.handleScenarioSave} cancelScenario={this.closeScenarioPopup}/>
                    </form>
                </Popup>
            </div>
        )
    }
}

ScenarioSetConfigForm.propTypes = {
    scenarioSet: PropTypes.object.isRequired,
    scenarios: PropTypes.array.isRequired,
    saveScenarioSet: PropTypes.func,
    createScenario: PropTypes.func,
    saveScenario: PropTypes.func,
    cancelScenarioSet: PropTypes.func
}

export default ScenarioSetConfigForm
