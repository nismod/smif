import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link } from 'react-router-dom'

import { fetchSosModelRuns, fetchSosModels, fetchSectorModels, fetchScenarioSets, fetchScenarios, fetchNarrativeSets, fetchNarratives } from '../../actions/actions.js'
import { createSosModelRun, createSosModel, createSectorModel, createScenarioSet, createScenario, createNarrativeSet, createNarrative } from '../../actions/actions.js'
import { saveSosModelRun,   saveSosModel,   saveSectorModel,   saveScenarioSet,   saveScenario,   saveNarrativeSet,   saveNarrative   } from '../../actions/actions.js'
import { deleteSosModelRun, deleteSosModel, deleteSectorModel, deleteScenarioSet, deleteScenario, deleteNarrativeSet, deleteNarrative } from '../../actions/actions.js'

import Popup from '../../components/ConfigForm/General/Popup.js'
import ProjectOverviewItem from '../../components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'
import CreateConfigForm from '../../components/ConfigForm/ProjectOverview/CreateConfigForm.js'
import DeleteForm from '../../components/ConfigForm/General/DeleteForm.js'

class ProjectOverview extends Component {
    constructor() {
        super()

        this.state = {
            createPopupIsOpen: false,
            createPopupType: 'none',
            deletePopupIsOpen: false,
            deletePopupConfigName: 'none',
            deletePopupType: 'none'
        }

        this.closeCreatePopup = this.closeCreatePopup.bind(this)
        this.openCreatePopup = this.openCreatePopup.bind(this)
        this.handleCreate = this.handleCreate.bind(this)

        this.closeDeletePopup = this.closeDeletePopup.bind(this)
        this.openDeletePopup = this.openDeletePopup.bind(this)
        this.handleDelete = this.handleDelete.bind(this)

        this.collectIdentifiers = this.collectIdentifiers.bind(this)

        this.handleInputChange = this.handleInputChange.bind(this)
    }

    componentDidMount () {
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
            createPopupType: event.target.name
        })
    }

    handleCreate(config) {

        const {createPopupType} = this.state
        const { dispatch } = this.props

        this.closeCreatePopup(createPopupType)
        
        switch(createPopupType) {
        case 'SosModelRun':
            dispatch(createSosModelRun(
                {
                    'name': config.name,
                    'description': config.description,
                    'stamp': new Date().toISOString(),
                    'sos_model': '',
                    'scenarios': {},
                    'narratives': {},
                    'decision_module': '',
                    'timesteps': [2015, 2020]
                }
            ))
            break
        case 'SosModel':
            dispatch(createSosModel(
                {
                    'name': config.name,
                    'description': config.description,
                    'sector_models': [],
                    'narrative_sets': [],
                    'scenario_sets': [],
                    'dependencies': [],
                    'max_iterations': 1,
                    'convergence_absolute_tolerance': "1e-05",
                    'convergence_relative_tolerance': "1e-05"
                }
            ))
            break
        case 'SectorModel':
            dispatch(createSectorModel(
                {
                    'name':  config.name,
                    'description': config.description,
                    'classname': '',
                    'path': '',
                    'inputs': [],
                    'outputs': [],
                    'parameters': [],
                    'interventions': [],
                    'initial_conditions': []
                }
            ))
            break
        case 'ScenarioSet':
            dispatch(createScenarioSet(
                {
                    'name':  config.name,
                    'description': config.description,
                    'facets': []
                }
            ))
            break
        case 'NarrativeSet':
            dispatch(createNarrativeSet(
                {
                    'name': config.name,
                    'description': config.description,
                }
            ))
            break
        case 'Narrative':
            dispatch(createNarrative(
                {
                    'name': config.name,
                    'description': config.description,
                    'narrative_set': '',
                    'filename': ''
                }
            ))
            break
        }
    }

    closeCreatePopup() {
        this.setState({createPopupIsOpen: false})
    }

    openDeletePopup(event) {
        
        let target_in_use_by = []

        switch(event.target.name) {
            case 'SosModel':
                this.props.sos_model_runs.forEach(function(sos_model_run) {   
                    if (sos_model_run.sos_model == event.target.value) {
                        target_in_use_by.push({
                            name: sos_model_run.name,
                            link: '/configure/sos-model-run/',
                            type: 'SosModelRun'
                        })
                    }                    
                })
                break

            case 'SectorModel':
                this.props.sos_models.forEach(function(sos_model) {   
                    
                    sos_model.sector_models.forEach(function(sector_model) {
                        if (sector_model == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })
                break

            case 'ScenarioSet':
                this.props.sos_models.forEach(function(sos_model) {   
                        
                    sos_model.scenario_sets.forEach(function(scenario_set) {
                        if (scenario_set == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })

                    sos_model.dependencies.forEach(function(dependency) {
                        if (dependency.source_model == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })
                break

            case 'NarrativeSet':
                this.props.sos_models.forEach(function(sos_model) {   
                        
                    sos_model.narrative_sets.forEach(function(narrative_set) {
                        if (narrative_set == event.target.value) {
                            target_in_use_by.push({
                                name: sos_model.name,
                                link: '/configure/sos-models/',
                                type: 'SosModel'
                            })
                        }
                    })
                })

                this.props.narratives.forEach(function(narrative) {   
                    if (narrative.narrative_set == event.target.value) {
                        target_in_use_by.push({
                            name: narrative.name,
                            link: '/configure/narratives/',
                            type: 'Narrative'
                        })
                    }
                })
                break

            case 'Narrative':
                this.props.sos_model_runs.forEach(function(sos_model_run) {   
                    Object.keys(sos_model_run.narratives).forEach(function(narrative_sets) {
                        sos_model_run.narratives[narrative_sets].forEach(function(narrative) {
                            if (narrative == event.target.value) {
                                target_in_use_by.push({
                                    name: sos_model_run.name,
                                    link: '/configure/sos-model-run/',
                                    type: 'SosModelRun'
                                })
                            }                    
                        })
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

    handleDelete() {

        const {deletePopupType, deletePopupConfigName} = this.state
        const { scenarios } = this.props
        const { dispatch } = this.props

        this.closeDeletePopup(deletePopupType)

        switch(deletePopupType) {
        case 'SosModelRun':
            dispatch(deleteSosModelRun(deletePopupConfigName))
            break
        case 'SosModel':
            dispatch(deleteSosModel(deletePopupConfigName))
            break
        case 'SectorModel':
            dispatch(deleteSectorModel(deletePopupConfigName))
            break
        case 'ScenarioSet':
            dispatch(deleteScenarioSet(deletePopupConfigName))
            break
        case 'NarrativeSet':
            dispatch(deleteNarrativeSet(deletePopupConfigName))
            break
        case 'Narrative':
            dispatch(deleteNarrative(deletePopupConfigName))
            break
        }
    }

    closeDeletePopup() {
        this.setState({deletePopupIsOpen: false})
    }

    collectIdentifiers() {
        const { sos_model_runs, sos_models, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching } = this.props
        let types = [sos_model_runs, sos_models, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching]

        let identifiers = []

        types.forEach(function(type){
            if (type.length > 0) {
                type.forEach(function(config){
                    identifiers.push(config.name)
                })
            }
        })

        return identifiers
    }

    render () {
        const { sos_model_runs, sos_models, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching } = this.props

        let used_identifiers = this.collectIdentifiers()

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
                            <input className="btn btn-secondary btn-lg btn-block" name="SosModelRun" type="button" value="Create a new Model Run" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            System-of-Systems Models
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="SosModel" items={sos_models} itemLink="/configure/sos-models/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="SosModel" type="button" value="Create a new System-of-Systems Model" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Simulation Model
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="SectorModel" items={sector_models} itemLink="/configure/sector-models/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="SectorModel" type="button" value="Create a new Simulation Model" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Scenario Sets
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="ScenarioSet" items={scenario_sets} itemLink="/configure/scenario-set/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="ScenarioSet" type="button" value="Create a new Scenario Set" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Narrative Sets
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="NarrativeSet" items={narrative_sets} itemLink="/configure/narrative-set/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="NarrativeSet" type="button" value="Create a new Narrative Set" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">
                            Narratives
                        </div>
                        <div className="card-body">
                            <ProjectOverviewItem itemname="Narrative" items={narratives} itemLink="/configure/narratives/" onDelete={this.openDeletePopup} />
                            <input className="btn btn-secondary btn-lg btn-block" name="Narrative" type="button" value="Create a new Narrative" onClick={this.openCreatePopup}/>
                        </div>
                    </div>

                    <br/>

                    {/* Popup for Create */}
                    <Popup onRequestOpen={this.state.createPopupIsOpen}>
                        <CreateConfigForm config_type={this.state.createPopupType} existing_names={used_identifiers} submit={this.handleCreate} cancel={this.closeCreatePopup}/>
                    </Popup>

                    {/* Popup for Delete */}
                    <Popup onRequestOpen={this.state.deletePopupIsOpen}>
                        <DeleteForm config_name={this.state.deletePopupConfigName} config_type={this.state.deletePopupType} in_use_by={this.state.deletePopupInUseBy} submit={this.handleDelete} cancel={this.closeDeletePopup}/>
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
