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
        const value = target.value
        const name = target.name

        if (name == 'Scenario') {
            this.setState({
                selectedScenarios: value
            })         
        } else {
            this.setState({
                selectedScenarioSet: update(this.state.selectedScenarioSet, {[name]: {$set: value}})
            })
        }
    }

    handleSave() {

        // delete all existing scenarios in this set
        let deleteScenarios = this.props.scenarios.filter(scenario => scenario.scenario_set == this.props.scenarioSet['name'])
        for (var i in deleteScenarios) {
            this.props.deleteScenario(deleteScenarios[i]['name'])
        }

        // add the current state of scenarios
        for (var i in this.state.selectedScenarios) {
            this.props.createScenario(this.state.selectedScenarios[i])
        }

        // save the scenario set
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

    handleFacetCreate(facet) {
        let {selectedScenarioSet} = this.state
        selectedScenarioSet.facets.push(facet)
        this.closeFacetPopup()
    }

    handleScenarioSave(saveScenario) {
        var newScenarios = []

        for (var i in this.state.selectedScenarios) {
            if (this.state.selectedScenarios[i]['name'] == saveScenario['name']) {
                newScenarios.push(saveScenario)
            } else {
                newScenarios.push(this.state.selectedScenarios[i])
            }
        }

        this.handleChange({target: {name: 'Scenario', value: newScenarios}})
        this.closeScenarioPopup()
    }

    handleScenarioCreate(scenario) {
        let {selectedScenarios} = this.state
        selectedScenarios.push(scenario)
        this.closeScenarioPopup()
    }

    handleCancel() {
        this.props.cancelScenarioSet()
    }

    openAddFacetPopup() {
        this.setState({selectedFacet: {name: undefined, description: ''}})
        this.setState({addFacetPopupIsOpen: true})
    }

    closeFacetPopup() {
        this.setState({addFacetPopupIsOpen: false})
    }

    openEditFacetPopup(id) {
        this.setState({selectedFacet: Object.assign({}, this.state.selectedScenarioSet.facets[id])})
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

        this.setState({selectedScenario: { name: undefined, description: '', 'facets': new_facets, 'scenario_set': this.state.selectedScenarioSet['name'] }})
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
        this.setState({selectedScenario: Object.assign({}, this.state.selectedScenarios[id])})
        this.setState({editScenarioPopupIsOpen: true})
    }

    renderScenarioSetConfigForm(selectedScenarioSet, selectedScenarios, selectedScenario, selectedFacet) {

        // Check if facets are configured in all scenario sets
        let scenarioWarnings = []
        
        let facetlist = []
        for (let facet of selectedScenarioSet.facets) {
            facetlist.push(facet['name'])
        }
        for (let scenario in selectedScenarios) {
            let scenarioFacetList = []
            for (let facet of selectedScenarios[scenario]['facets']) {
                scenarioFacetList.push(facet['name'])
            }

            if (facetlist.toString() == scenarioFacetList.toString()) {
                scenarioWarnings.push(false)
            } else {
                scenarioWarnings.push(true)
            }
        }

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
                        <input className="btn btn-secondary btn-lg btn-block" name="createFacet" type="button" value="Add Facet" onClick={this.openAddFacetPopup}/>
                    </div>
                </div>

                <br/>

                <div className="card">
                    <div className="card-header">Scenarios</div>
                    <div className="card-body">
                        <PropertyList itemsName="Scenario" items={selectedScenarios} columns={{name: 'Name', description: 'Description'}} enableWarnings={true} rowWarning={scenarioWarnings} editButton={true} deleteButton={true} onEdit={this.openEditScenarioPopup} onDelete={this.handleChange} />
                        <input className="btn btn-secondary btn-lg btn-block" name="createScenario" type="button" value="Add Scenario" onClick={this.openAddScenarioPopup}/>
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
                    <form onSubmit={(e) => {e.preselectedScenarioventDefault(); e.stopPropagation()}}>                
                        <ScenarioConfigForm scenario={selectedScenario} scenarioSet={selectedScenarioSet} createScenario={this.handleScenarioCreate} saveScenario={this.handleScenarioSave} cancelScenario={this.closeScenarioPopup}/>
                    </form>
                </Popup>
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div>
                <div id="alert-danger" className="alert alert-danger">
                    {message}
                </div>
                <div>
                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
                </div>
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div>
                <div id="alert-warning" className="alert alert-warning">
                    {message}
                
                </div>
                <div>
                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
                </div>
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div>
                <div id="alert-info" className="alert alert-info">
                    {message}
                
                </div>
                <div>
                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
                </div>
            </div>
        )
    }

    render() {
        const {selectedScenarioSet, selectedScenarios, selectedScenario, selectedFacet} = this.state
        
        if (selectedScenarioSet == null || selectedScenarioSet == undefined) {
            return this.renderDanger('The selectedScenarioSet are not initialised')
        } else if (selectedScenarioSet.facets == null || selectedScenarioSet.facets == undefined) {
            return this.renderDanger('The selectedScenarioSet.facets is not initialised')
        } else if (selectedScenarios == null || selectedScenarios == undefined) {
            return this.renderDanger('The selectedScenarios is not initialised')
        } else if (selectedScenario == null || selectedScenario == undefined) {
            return this.renderDanger('The selectedScenario is not initialised')
        } else if (selectedFacet == null || selectedFacet == undefined) {
            return this.renderDanger('The selectedFacet is not initialised')
        } else {
            return this.renderScenarioSetConfigForm(selectedScenarioSet, selectedScenarios, selectedScenario, selectedFacet)
        }
    }
}

ScenarioSetConfigForm.propTypes = {
    scenarioSet: PropTypes.object.isRequired,
    scenarios: PropTypes.array.isRequired,
    saveScenarioSet: PropTypes.func,
    createScenario: PropTypes.func,
    saveScenario: PropTypes.func,
    deleteScenario: PropTypes.func,
    cancelScenarioSet: PropTypes.func
}

export default ScenarioSetConfigForm
