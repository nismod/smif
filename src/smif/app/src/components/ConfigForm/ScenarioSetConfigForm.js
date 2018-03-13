import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from './General/Popup.js'
import PropertyList from './General/PropertyList.js'
import ScenarioConfigForm from './ScenarioSet/ScenarioConfigForm.js'
import FacetConfigForm from './ScenarioSet/FacetConfigForm.js'
import DeleteForm from '../../components/ConfigForm/General/DeleteForm.js'

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

        this.openDeletePopup = this.openDeletePopup.bind(this)
        this.closeDeletePopup = this.closeDeletePopup.bind(this)
        this.deletePopupSubmit = this.deletePopupSubmit.bind(this)

        this.state = {
            scenarios: this.props.scenarios,
            selectedFacet: {},
            selectedScenario: {},
            selectedScenarioSet: this.props.scenarioSet,
            selectedScenarios: this.props.scenarios.filter(scenario => scenario.scenario_set == this.props.scenarioSet.name),
            addFacetPopupIsOpen: false,
            editScenarioPopupIsOpen: false,
            deletePopupIsOpen: false
        }
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
        this.props.saveScenario(saveScenario)
        this.closeScenarioPopup()
    }

    handleScenarioCreate(scenario) {
        let {selectedScenarios} = this.state
        this.props.createScenario(scenario)
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

    openEditFacetPopup(name) {
        const {selectedScenarioSet} = this.state

        // Get id
        let id
        for (let i = 0; i < selectedScenarioSet.facets.length; i++) {
            if (selectedScenarioSet.facets[i].name == name) {
                id = i
            }
        }

        this.setState({selectedFacet: Object.assign({}, selectedScenarioSet.facets[id])})
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
    
    openEditScenarioPopup(name) {

        const { selectedScenarios, selectedScenarioSet} = this.state

        // Get id
        let id
        for (let i = 0; i < selectedScenarios.length; i++) {
            if (selectedScenarios[i].name == name) {
                id = i
            }
        }

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

    openDeletePopup(event) {
        
        let target_in_use_by = []

        switch(event.target.name) {
            case 'Facet':
                this.props.sosModels.forEach(function(sos_model) {   
                    sos_model.dependencies.forEach(function(dependency) {
                        if (event.target.value == dependency.source_model_output) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })
                break

            case 'Scenario':
                this.props.sosModelRuns.forEach(function(sos_model_run) {   
                    Object.keys(sos_model_run.scenarios).forEach(function(key) {
                        if (event.target.value == sos_model_run.scenarios[key]) {
                            target_in_use_by.push({
                                name: sos_model_run.name,
                                link: '/configure/sos-model-run/',
                                type: 'SosModelRun'
                            })
                        }
                    })
                })
                break
        }

        this.setState({
            deletePopupIsOpen: true,
            deletePopupConfigName: event.target.value,
            deletePopupType: event.target.name,
            deletePopupInUseBy: target_in_use_by
        })
    }

    deletePopupSubmit() {

        const {deletePopupType, deletePopupConfigName, selectedScenarioSet, selectedScenarios} = this.state
        const { scenarios } = this.props
        const { dispatch } = this.props

        switch(deletePopupType) {
            case 'Facet':
                for (let i = 0; i < Object.keys(selectedScenarioSet.facets).length; i++) {
                    if (selectedScenarioSet.facets[i].name == deletePopupConfigName) {
                        selectedScenarioSet.facets.splice(i, 1)
                    }
                }
                break
                
            case 'Scenario':
                this.props.deleteScenario(deletePopupConfigName)
                break
        }

        this.closeDeletePopup(deletePopupType)
        this.forceUpdate()
    }

    closeDeletePopup() {
        this.setState({deletePopupIsOpen: false})
    }

    renderScenarioSetConfigForm(selectedScenarioSet, selectedScenarios, selectedScenario, selectedFacet) {

        // Do not show scenarios when there are no facets configured
        let scenarioCardState = 'collapse show'
        if (selectedScenarioSet.facets.length == 0) {
            scenarioCardState = 'collapse'
        }

        // Check if facets are configured in all scenario sets
        // prepare an array with warning
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
                        <PropertyList itemsName="Facet" items={selectedScenarioSet.facets} columns={{name: 'Name', description: 'Description'}} editButton={true} deleteButton={true} onEdit={this.openEditFacetPopup} onDelete={this.openDeletePopup} />
                        <input className="btn btn-secondary btn-lg btn-block" name="createFacet" type="button" value="Add Facet" onClick={this.openAddFacetPopup}/>
                    </div>
                </div>

                <br/>

                <div className={scenarioCardState} >
                    <div className="card">
                        <div className="card-header">Scenarios</div>
                        <div className="card-body">
                            <PropertyList itemsName="Scenario" items={selectedScenarios} columns={{name: 'Name', description: 'Description'}} enableWarnings={true} rowWarning={scenarioWarnings} editButton={true} deleteButton={true} onEdit={this.openEditScenarioPopup} onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createScenario" type="button" value="Add Scenario" onClick={this.openAddScenarioPopup}/>
                        </div>
                    </div>
                </div>

                <Popup onRequestOpen={this.state.deletePopupIsOpen}>
                    <DeleteForm config_name={this.state.deletePopupConfigName} config_type={this.state.deletePopupType} in_use_by={this.state.deletePopupInUseBy} submit={this.deletePopupSubmit} cancel={this.closeDeletePopup}/>
                </Popup>

                <br/>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSave} />
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
    sosModelRuns: PropTypes.array.isRequired, 
    sosModels: PropTypes.array.isRequired,
    scenarioSet: PropTypes.object.isRequired,
    scenarios: PropTypes.array.isRequired,
    saveScenarioSet: PropTypes.func,
    createScenario: PropTypes.func,
    saveScenario: PropTypes.func,
    deleteScenario: PropTypes.func,
    cancelScenarioSet: PropTypes.func
}

export default ScenarioSetConfigForm
