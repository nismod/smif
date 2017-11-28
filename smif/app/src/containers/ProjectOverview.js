import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { connect } from 'react-redux';
import { Link } from 'react-router-dom';

// import Modal from 'react-modal';

import { fetchSosModelRuns } from '../actions/actions.js';
import { fetchSosModels } from '../actions/actions.js';

import { createSosModelRun } from '../actions/actions.js'
import { deleteSosModelRun } from '../actions/actions.js'

import Popup from '../components/Popup.js'
import ProjectOverviewItem from '../components/ProjectOverviewItem.js';

class ProjectOverview extends Component {
    constructor() {
        super();

        this.handleInputChange = this.handleInputChange.bind(this)
        this.createSosModelRun = this.createSosModelRun.bind(this)
        this.deleteSosModelRun = this.deleteSosModelRun.bind(this)

        this.handleProjectSave = this.handleProjectSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            CreateSosModelRunpopupIsOpen: false,
            CreateSosModelpopupIsOpen: false
        }
        this.closeCreateSosModelRunPopup = this.closeCreateSosModelRunPopup.bind(this)
        this.openCreateSosModelRunPopup = this.openCreateSosModelRunPopup.bind(this)
        this.closeCreateSosModelPopup = this.closeCreateSosModelPopup.bind(this)
        this.openCreateSosModelPopup = this.openCreateSosModelPopup.bind(this)
    }

    componentWillMount () {
        const { dispatch } = this.props;
        dispatch(fetchSosModelRuns());
        dispatch(fetchSosModels());
    }

    handleInputChange(event) {
        
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name
        
        this.setState({
            [name]: value
        });
    }

    openCreateSosModelRunPopup() {
        this.setState({CreateSosModelRunpopupIsOpen: true});
    }
    
    closeCreateSosModelRunPopup() {
        this.setState({CreateSosModelRunpopupIsOpen: false});
    }

    openCreateSosModelPopup() {
        this.setState({CreateSosModelpopupIsOpen: true});
    }
    
    closeCreateSosModelPopup() {
        this.setState({CreateSosModelpopupIsOpen: false});
    }
    
    createSosModelRun() {
        const { dispatch } = this.props
        
        this.closeCreateSosModelRunPopup()

        dispatch(createSosModelRun(this.state.newSosModelRun_name))
        dispatch(fetchSosModelRuns())
    }
    
    deleteSosModelRun(sosModelRunName) {
        const { dispatch } = this.props;
        dispatch(deleteSosModelRun(sosModelRunName))
        dispatch(fetchSosModelRuns())
    }

    deleteSosModel(sosModelName) {
        return null
    }

    handleProjectSave() {
        console.log(this.props)
    }

    handleCancel() {
        console.log(this.state)
    }

    render () {
        const { sos_model_runs, sos_models, isFetching } = this.props;

        return (
            <div>
                <h1>Project Overview</h1>

                <div hidden={ !isFetching } className="alert alert-primary">
                    Loading...
                </div>

                <div hidden className="alert alert-danger">
                    Error
                </div>

                <div hidden={ isFetching }>
                    <label>
                        Project name
                        <input name="project_name" type="text" defaultValue="NISMOD v2.0"/>
                    </label>

                    <h2>Model Runs</h2>
                    <ProjectOverviewItem items={sos_model_runs} itemLink="/configure/sos-model-run/" onDelete={this.deleteSosModelRun} />
                    <input type="button" value="Create a new Model Run" onClick={this.openCreateSosModelRunPopup}/>
                    <Popup onRequestOpen={this.state.CreateSosModelRunpopupIsOpen}>
                        <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.createSosModelRun();}}>
                            <h2 ref={subtitle => this.subtitle = subtitle}>Create a new Model Run</h2>
                            <input name="newSosModelRun_name" type="text" onChange={this.handleInputChange}/>
                            <input type="submit" value="Create"/>
                        </form>
                        <input type="button" value="Cancel" onClick={this.closeCreateSosModelRunPopup}/>
                    </Popup>

                    <h2>System-of-Systems Models</h2>
                    <ProjectOverviewItem items={sos_models} itemLink="/configure/sos-models/" onDelete={this.deleteSosModel} />
                    <input type="button" value="Create a new System-of-Systems Configuration" onClick={this.openCreateSosModelPopup}/>
                    <Popup onRequestOpen={this.state.CreateSosModelpopupIsOpen}>
                        <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.createSosModel();}}>
                            <h2 ref={subtitle => this.subtitle = subtitle}>Create a new Model</h2>
                            <input name="newSosModel_name" type="text" onChange={this.handleInputChange}/>
                            <input type="submit" value="Create"/>
                        </form>
                        <input type="button" value="Cancel" onClick={this.closeCreateSosModelPopup}/>
                    </Popup>

                    <h2>Simulation Models</h2>
                    <div className="select-container">
                        <select>
                            <option>Energy Demand</option>
                            <option>Energy Supply</option>
                            <option>Water</option>
                            <option>Transport</option>
                            <option>Solid Waste</option>
                        </select>
                    </div>
                    <input type="button" value="Edit/View Simulation Model Configuration" />
                    <input type="button" value="Create a new Simulation Model Configuration" />

                    <h2>Scenarios</h2>
                    <input type="button" value="Create a Scenario" />

                    <h2>Narratives</h2>
                    <input type="button" value="Create a Narrative" />

                    <input type="button" value="Save Project Configuration" onClick={this.handleProjectSave} />
                    <input type="button" value="Cancel" onClick={this.handleCancel} />
                </div>
            </div>
        );
    }
}

ProjectOverview.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    sos_models: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
};

function mapStateToProps(state) {
    const { sos_model_runs, sos_models } = state;

    return {
        sos_model_runs: sos_model_runs.items,
        sos_models: sos_models.items,
        isFetching: (sos_models.isFetching && sos_model_runs.isFetching)
    };
}

export default connect(mapStateToProps)(ProjectOverview);
