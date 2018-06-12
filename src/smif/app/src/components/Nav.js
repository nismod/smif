import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { NavLink, Route } from 'react-router-dom';
import Footer from '../containers/Footer'
import {fetchSosModelRuns, fetchSosModels, fetchSectorModels, fetchScenarioSets, fetchScenarios, fetchNarrativeSets, fetchNarratives} from '../actions/actions.js'

import {FaHome, FaSliders, FaSitemap, FaCode, FaBarChart} from 'react-icons/lib/fa'
import { Badge, Button } from 'reactstrap'

class Nav extends Component {
    constructor(props) {
        super(props)
        this.init = true
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

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderError() {
        return (
            <div className="alert alert-danger">
                Error
            </div>
        )
    }

    renderNav(sos_model_runs, sos_models, sector_models, scenario_sets, narrative_sets, narratives) {
        return (
            <div>
                <nav className="col-md-2 d-none d-md-block bg-light sidebar">
                <div className="sidebar-sticky">
                    <ul className="nav flex-column">
                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/">
                            <FaHome size={20}/>
                            <span className="ml-1">Welcome</span>
                        </NavLink>
                    </li>
                    </ul>

                    <h6 className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
                    <span>Simulation</span>
                    </h6>
                    <ul className="nav flex-column mb-2">
                    <li className="nav-item">
                        <NavLink className="nav-link" to="/configure/sos-model-run" >
                            <FaSliders size={20}/>
                            <span className="ml-1">Model Runs <Badge color="secondary">{sos_model_runs.length}</Badge></span>
                        </NavLink>
                        <Route path="/configure/sos-model-run/" render={() => 
                            <div>
                                {sos_model_runs.map(sos_model_run =>
                                    <div key={'nav_' + sos_model_run.name}> 
                                        <NavLink className="nav-link" to={"/configure/sos-model-run/" + sos_model_run.name} >
                                            <span className="ml-3"/>
                                            <span className="ml-1 text-muted">{sos_model_run.name}</span>
                                        </NavLink>
                                    </div>
                                )}
                            </div>
                        }/>
                    </li>
                    </ul>

                    <h6 className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
                    <span>Configuration</span>
                    </h6>

                    <ul className="nav flex-column mb-2">
                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/configure/sos-models" >
                            <FaSitemap size={20}/>
                            <span className="ml-1">Models <Badge color="secondary">{sos_models.length}</Badge></span>
                        </NavLink>
                        <Route path="/configure/sos-models/" render={() => 
                            <div>
                            {sos_models.map(sos_models => 
                                <div key={'nav_' + sos_models.name}>
                                    <NavLink exact className="nav-link" to={"/configure/sos-models/" + sos_models.name} >
                                        <span className="ml-3"/>
                                        <span className="ml-1 text-muted">{sos_models.name}</span>
                                    </NavLink>
                                </div>
                            )}
                            </div>
                        }/>

                        <NavLink exact className="nav-link" to="/configure/sector-models" >
                            <FaCode size={20}/>
                            <span className="ml-1">Wrappers <Badge color="secondary">{sector_models.length}</Badge></span>
                        </NavLink>
                        <Route path="/configure/sector-models/" render={() => 
                            <div>
                            {sector_models.map(sector_models =>
                                <div key={'nav_' + sector_models.name}>
                                    <NavLink exact className="nav-link" to={"/configure/sector-models/" + sector_models.name} >
                                        <span className="ml-3"/>
                                        <span className="ml-1 text-muted">{sector_models.name}</span>
                                    </NavLink>
                                </div>
                            )}
                            </div>
                        }/>            
                    </li>

                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/configure/scenario-set" >
                            <FaBarChart size={20}/>
                            <span className="ml-1">Scenarios <Badge color="secondary">{scenario_sets.length}</Badge></span>
                        </NavLink>
                        <Route path="/configure/scenario-set/" render={() => 
                            <div>
                            {scenario_sets.map(config =>
                                <div key={'nav_' + config.name}>
                                    <NavLink exact className="nav-link" to={"/configure/scenario-set/" + config.name} >
                                        <span className="ml-3"/>
                                        <span className="ml-1 text-muted">{config.name}</span>
                                    </NavLink>
                                </div>
                            )}
                            </div>
                        }/>
                    </li>

                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/configure/narrative-set" >
                            <FaBarChart size={20}/>
                            <span className="ml-1">Narratives <Badge color="secondary">{narrative_sets.length}</Badge></span>
                        </NavLink>    

                        <Route path="/configure/narrative*" render={() => 
                            <div>
                            <span className="ml-4"/>
                            <span className="ml-1">Sets</span>
                            {narrative_sets.map(config =>
                                <div key={'nav_' + config.name}>
                                    <NavLink exact className="nav-link" to={"/configure/narrative-set/" + config.name} >
                                        <span className="ml-3"/>
                                        <span className="ml-1 text-muted">{config.name}</span>
                                    </NavLink>
                                </div>
                            )}
                            </div>
                        }/>
                        <Route path="/configure/narrative*" render={() => 
                            <div>
                            <span className="ml-4"/>
                            <span className="ml-1">Data</span>
                            {narratives.map(config =>
                                <div key={'nav_' + config.name}>
                                    <NavLink exact className="nav-link" to={"/configure/narratives/" + config.name} >
                                        <span className="ml-3"/>
                                        <span className="ml-1 text-muted">{config.name}</span>
                                    </NavLink>
                                </div>
                            )}
                            </div>
                        }/>
                    </li>
                    </ul>
                </div>
                <Footer />
            </nav>
            </div>
        )
    }

    render() {
        const {sos_model_runs, sos_models, sector_models, scenario_sets, narrative_sets, narratives, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderNav(sos_model_runs, sos_models, sector_models, scenario_sets, narrative_sets, narratives)
        }
    }
}

Nav.propTypes = {
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

        isFetching: (
            state.sos_model_runs.isFetching || 
            state.sos_models.isFetching || 
            state.sector_models.isFetching || 
            state.scenario_sets.isFetching || 
            state.scenarios.isFetching || 
            state.narrative_sets.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(Nav)
