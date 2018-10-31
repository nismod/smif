import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { NavLink, Route } from 'react-router-dom'
import Footer from 'containers/Footer'
import { fetchModelRuns, fetchSosModels, fetchSectorModels, fetchScenarios } from 'actions/actions.js'

import {FaHome, FaTasks, FaSliders, FaSitemap, FaCode, FaBarChart} from 'react-icons/lib/fa'
import { Badge } from 'reactstrap'

class Nav extends Component {
    constructor(props) {
        super(props)
        this.init = true
    }

    componentDidMount () {
        const { dispatch } = this.props

        dispatch(fetchModelRuns())
        dispatch(fetchSosModels())
        dispatch(fetchSectorModels())
        dispatch(fetchScenarios())
    }

    renderLoading() {
        return (
            <nav className="col-12 col-md-3 col-xl-2 bg-light sidebar">
                <ul className="nav flex-column">
                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/">
                            <FaHome size={20}/>
                            Home
                        </NavLink>
                    </li>
                </ul>
                <div className="alert alert-primary">
                    Loading...
                </div>
            </nav>
        )
    }

    renderError() {
        return (
            <nav className="col-12 col-md-3 col-xl-2 bg-light sidebar">
                <ul className="nav flex-column">
                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/">
                            <FaHome size={20}/>
                            Home
                        </NavLink>
                    </li>
                </ul>
                <div className="alert alert-danger">
                    Error
                </div>
            </nav>
        )
    }

    renderNav(model_runs, sos_models, sector_models, scenarios, narratives) {
        var job_status = ['unstarted', 'running', 'stopped', 'done', 'failed']

        return (
            <nav className="col-12 col-md-3 col-xl-2 bg-light sidebar">
                <ul className="nav flex-column">
                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/">
                            <FaHome size={20}/>
                            Home
                        </NavLink>
                    </li>
                </ul>

                <h6 className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
                    <span>Simulation</span>
                </h6>
                <ul className="nav flex-column mb-2">

                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/jobs/" >
                            <FaTasks size={20}/>
                        Jobs
                            <Badge color="secondary">{model_runs.length}</Badge>
                        </NavLink>
                        <Route path="/jobs/" render={() =>
                            <ul className="nav flex-column">
                                {job_status.map(status =>
                                    <li key={'nav_' + status} className="nav-item">
                                        <NavLink
                                            className="nav-link"
                                            to={'/jobs/status=' + status} >
                                            {status}
                                        </NavLink>
                                    </li>
                                )}
                            </ul>
                        }/>
                    </li>

                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/configure/model-runs" >
                            <FaSliders size={20}/>
                        Model Runs
                            <Badge color="secondary">{model_runs.length}</Badge>
                        </NavLink>
                        <Route path="/configure/model-runs/" render={() =>
                            <ul className="nav flex-column">
                                {model_runs.map(model_run =>
                                    <li key={'nav_modelrun_' + model_run.name} className="nav-item">
                                        <NavLink
                                            key={'nav_' + model_run.name}
                                            className="nav-link"
                                            to={'/configure/model-runs/' + model_run.name} >
                                            {model_run.name}
                                        </NavLink>
                                    </li>
                                )}
                            </ul>
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
                        System-of-Systems Models
                            <Badge color="secondary">{sos_models.length}</Badge>
                        </NavLink>
                        <Route path="/configure/sos-models/" render={() =>
                            <ul className="nav flex-column">
                                {sos_models.map(sos_model =>
                                    <li key={'nav_sosmodel_' + sos_model.name} className="nav-item">
                                        <NavLink
                                            key={'nav_' + sos_model.name}
                                            exact
                                            className="nav-link"
                                            to={'/configure/sos-models/' + sos_model.name} >
                                            {sos_model.name}
                                        </NavLink>
                                    </li>
                                )}
                            </ul>
                        }/>

                        <NavLink exact className="nav-link" to="/configure/sector-models" >
                            <FaCode size={20}/>
                        Model Wrappers
                            <Badge color="secondary">{sector_models.length}</Badge>
                        </NavLink>
                        <Route path="/configure/sector-models/" render={() =>
                            <ul className="nav flex-column">
                                {sector_models.map(sector_model =>
                                    <li key={'nav_sectormodel_' + sector_model.name} className="nav-item">
                                        <NavLink
                                            key={'nav_' + sector_model.name}
                                            exact
                                            className="nav-link"
                                            to={'/configure/sector-models/' + sector_model.name} >
                                            {sector_model.name}
                                        </NavLink>
                                    </li>
                                )}
                            </ul>
                        }/>
                    </li>

                    <li className="nav-item">
                        <NavLink exact className="nav-link" to="/configure/scenarios" >
                            <FaBarChart size={20}/>
                        Scenarios
                            <Badge color="secondary">{scenarios.length}</Badge>
                        </NavLink>
                        <Route path="/configure/scenarios/" render={() =>
                            <ul className="nav flex-column">
                                {scenarios.map(scenario =>
                                    <li key={'nav_scenario_' + scenario.name} className="nav-item">
                                        <NavLink
                                            key={'nav_' + scenario.name}
                                            exact
                                            className="nav-link"
                                            to={'/configure/scenarios/' + scenario.name} >
                                            {scenario.name}
                                        </NavLink>
                                    </li>
                                )}
                            </ul>
                        }/>
                    </li>
                </ul>
                <Footer />
            </nav>
        )
    }

    render() {
        const {model_runs, sos_models, sector_models, scenarios, isFetching} = this.props
        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderNav(model_runs, sos_models, sector_models, scenarios)
        }
    }
}

Nav.propTypes = {
    model_runs: PropTypes.array.isRequired,
    sos_models: PropTypes.array.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
}

function mapStateToProps(state) {
    const { model_runs, sos_models, sector_models, scenarios } = state
    return {
        model_runs: model_runs.items,
        sos_models: sos_models.items,
        sector_models: sector_models.items,
        scenarios: scenarios.items,

        isFetching: (
            state.model_runs.isFetching ||
            state.sos_models.isFetching ||
            state.sector_models.isFetching ||
            state.scenarios.isFetching
        )
    }
}

export default connect(mapStateToProps)(Nav)
