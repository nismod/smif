import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link } from 'react-router-dom'

import { fetchSosModelRuns, fetchSosModels, fetchSectorModels, fetchScenarioSets, fetchScenarios, fetchNarrativeSets, fetchNarratives } from '../../actions/actions.js'
import { createSosModelRun, createSosModel, createSectorModel, createScenarioSet, createScenario, createNarrativeSet, createNarrative } from '../../actions/actions.js'
import { deleteSosModelRun, deleteSosModel, deleteSectorModel, deleteScenarioSet, deleteScenario, deleteNarrativeSet, deleteNarrative } from '../../actions/actions.js'

import Popup from '../../components/ConfigForm/General/Popup.js'
import ProjectOverviewItem from '../../components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'

class ProjectOverview extends Component {
    constructor() {
        super()

        this.state = {
            createPopupIsOpen: false,
            createPopupHeader: 'none',
            createPopupType: 'none',
            deletePopupIsOpen: false,
            deletePopupHeader: 'none',
            deletePopupType: 'none'
        }

        this.closeCreatePopup = this.closeCreatePopup.bind(this)
        this.openCreatePopup = this.openCreatePopup.bind(this)
        this.createPopupSubmit = this.createPopupSubmit.bind(this)

        this.closeDeletePopup = this.closeDeletePopup.bind(this)
        this.openDeletePopup = this.openDeletePopup.bind(this)
        this.deletePopupSubmit = this.deletePopupSubmit.bind(this)

        this.handleInputChange = this.handleInputChange.bind(this)
    }

    componentWillMount () {
        const { dispatch } = this.props
        dispatch(fetchSosModelRuns())
        dispatch(fetchSosModels())
        dispatch(fetchSectorModels())
        dispatch(fetchScenarioSets())
        dispatch(fetchScenarios())
        dispatch(fetchNarrativeSets())
        dispatch(fetchNarratives())
    }

    handleInputChange(event) {

        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            [name]: value
        })
    }

    openCreatePopup(event) {

        this.setState({
            createPopupIsOpen: true,
            createPopupHeader: event.target.value,
            createPopupType: event.target.name
        })
    }

    createPopupSubmit() {

        const {createPopupType, createPopupName} = this.state
        const { dispatch } = this.props

        this.closeCreatePopup(createPopupType)

        switch(createPopupType) {
        case 'createSosModelRun':
            dispatch(createSosModelRun(createPopupName))
            dispatch(fetchSosModelRuns())
            break
        case 'createSosModel':
            dispatch(createSosModel(createPopupName))
            dispatch(fetchSosModels())
            break
        case 'createSectorModel':
            dispatch(createSectorModel(createPopupName))
            dispatch(fetchSectorModels())
            break
        case 'createScenarioSet':
            dispatch(createScenarioSet(createPopupName))
            dispatch(fetchScenarioSets())
            break
        case 'createScenario':
            dispatch(createScenario(createPopupName))
            dispatch(fetchScenarios())
            break
        case 'createNarrativeSet':
            dispatch(createNarrativeSet(createPopupName))
            dispatch(fetchNarrativeSets())
            break
        case 'createNarrative':
            dispatch(createNarrative(createPopupName))
            dispatch(fetchNarratives())
            break
        }
    }

    closeCreatePopup() {
        this.setState({createPopupIsOpen: false})
    }

    openDeletePopup(event) {
        this.setState({
            deletePopupIsOpen: true,
            deletePopupHeader: event.target.value,
            deletePopupType: event.target.name
        })
    }

    deletePopupSubmit() {

        const {deletePopupType, deletePopupHeader} = this.state
        const { dispatch } = this.props

        this.closeDeletePopup(deletePopupType)

        switch(deletePopupType) {
        case 'deleteSosModelRun':
            dispatch(deleteSosModelRun(deletePopupHeader))
            dispatch(fetchSosModelRuns())
            break
        case 'deleteSosModel':
            dispatch(deleteSosModel(deletePopupHeader))
            dispatch(fetchSosModels())
            break
        case 'deleteSectorModel':
            dispatch(deleteSectorModel(deletePopupHeader))
            dispatch(fetchSectorModels())
            break
        case 'deleteScenarioSet':
            dispatch(deleteScenarioSet(deletePopupHeader))
            dispatch(fetchScenarioSets())
            break
        case 'deleteScenario':
            dispatch(deleteScenario(deletePopupHeader))
            dispatch(fetchScenarios())
            break
        case 'deleteNarrativeSet':
            dispatch(deleteNarrativeSet(deletePopupHeader))
            dispatch(fetchNarrativeSets())
            break
        case 'deleteNarrative':
            dispatch(deleteNarrative(deletePopupHeader))
            dispatch(fetchNarratives())
            break
        }
    }

    closeDeletePopup() {
        this.setState({deletePopupIsOpen: false})
    }

    render () {
        const { sos_model_runs, sos_models, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching } = this.props

        return (
            <div>
                <div hidden={ !isFetching } className="alert alert-primary">
                    Loading...
                </div>

                <div hidden className="alert alert-danger">
                    Error
                </div>

                <div hidden={ isFetching }>
                    <div className="card">
                        <div className="card-header">
                            Project information
                        </div>
                        <div className="card-body">
                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" type="text" defaultValue="NISMOD v2.0"/>
                                </div>
                            </div>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Model Runs
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="SosModelRun" items={sos_model_runs} itemLink="/configure/sos-model-run/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createSosModelRun" type="button" value="Create a new Model Run" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            System-of-Systems Models
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="SosModel" items={sos_models} itemLink="/configure/sos-models/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createSosModel" type="button" value="Create a new System-of-Systems Model" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Simulation Model
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="SectorModel" items={sector_models} itemLink="/configure/sector-models/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createSectorModel" type="button" value="Create a new Simulation Model" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Scenario Sets
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="ScenarioSet" items={scenario_sets} itemLink="/configure/scenario-set/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createScenarioSet" type="button" value="Create a new Scenario Set" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Scenarios
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="Scenario" items={scenarios} itemLink="/configure/scenarios/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createScenario" type="button" value="Create a new Scenario" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Narrative Sets
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="NarrativeSet" items={narrative_sets} itemLink="/configure/narrative-set/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createNarrativeSet" type="button" value="Create a new Narrative Set" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Narratives
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="Narrative" items={narratives} itemLink="/configure/narratives/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="createNarrative" type="button" value="Create a new Narrative" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    {/* Popup for Create */}
                    <Popup onRequestOpen={this.state.createPopupIsOpen}>
                        <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.createPopupSubmit()}}>
                            <h2 ref={subtitle => this.subtitle = subtitle}>{this.state.createPopupHeader}</h2>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input autoFocus className="form-control" name="createPopupName" type="text" onChange={this.handleInputChange} required/>
                                </div>
                            </div>

                            <input className="btn btn-secondary btn-lg btn-block" type="submit" value="Create"/>
                            <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeCreatePopup}/>
                        </form>
                    </Popup>

                    {/* Popup for Delete */}
                    <Popup onRequestOpen={this.state.deletePopupIsOpen}>
                        <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.deletePopupSubmit()}}>
                            <h2 ref={subtitle => this.subtitle = subtitle}>Confirm delete</h2>
                            Are you sure you would like to delete {this.state.deletePopupHeader}?
                            <br/>
                            <input autoFocus className="btn btn-secondary btn-lg btn-block" type="submit" value="Delete"/>
                            <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeDeletePopup}/>
                        </form>
                    </Popup>

                </div>
            </div>
        )
    }
}

ProjectOverview.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    sos_models: PropTypes.array.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenario_sets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrative_sets: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
}

function mapStateToProps(state) {
    const { sos_model_runs, sos_models, sector_models, scenario_sets, scenarios, narrative_sets, narratives } = state

    return {
        sos_model_runs: sos_model_runs.items,
        sos_models: sos_models.items,
        sector_models: sector_models.items,
        scenario_sets: scenario_sets.items,
        scenarios: scenarios.items,
        narrative_sets: narrative_sets.items,
        narratives: narratives.items,
        isFetching: false
    }
}

export default connect(mapStateToProps)(ProjectOverview)
