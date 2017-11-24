import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { connect } from 'react-redux';
import { Link } from 'react-router-dom';

import Modal from 'react-modal';

import { fetchSosModelRuns } from '../actions/actions.js';
import { createSosModelRun } from '../actions/actions.js'
import { deleteSosModelRun } from '../actions/actions.js'

import SosModelRunItem from '../components/SosModelRunItem.js';

const customStyles = {
    content : {
        top                   : '50%',
        left                  : '50%',
        right                 : 'auto',
        bottom                : 'auto',
        marginRight           : '-50%',
        transform             : 'translate(-50%, -50%)'
    }
};

class ProjectOverview extends Component {
    constructor() {
        super();

        this.handleInputChange = this.handleInputChange.bind(this)
        this.createSosModelRun = this.createSosModelRun.bind(this)
        this.deleteSosModelRun = this.deleteSosModelRun.bind(this)

        this.state = {
            CreateSosModelRunpopupIsOpen: false
        }
    
        this.openCreateSosModelRunPopup = this.openCreateSosModelRunPopup.bind(this)
        this.closeCreateSosModelRunPopup = this.closeCreateSosModelRunPopup.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props;
        dispatch(fetchSosModelRuns());
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
        console.log('hello')
        this.setState({CreateSosModelRunpopupIsOpen: false});
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

    render () {
        const { sos_model_runs, isFetching } = this.props;
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
                    <table className="table table-sm">
                        <thead className="thead-light">
                            <tr>
                                <th scope="col">Name</th>
                                <th scope="col">Description</th>
                                <th scope="col"></th>
                                <th scope="col"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {
                                sos_model_runs.map((sos_model_run) => (
                                    <SosModelRunItem key={sos_model_run.name} onDelete={this.deleteSosModelRun}
                                        {...sos_model_run} />
                                ))
                            }
                        </tbody>
                    </table>
                    <input type="button" value="Create a new Model Run" onClick={this.openCreateSosModelRunPopup}/>
                    <Modal isOpen={this.state.CreateSosModelRunpopupIsOpen} onRequestClose={this.closeCreateSosModelRunPopup} style={customStyles} contentLabel="Example CreateSosModelRunPopup">   
                        <div>
                            <form onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.createSosModelRun();}}>
                                <h2 ref={subtitle => this.subtitle = subtitle}>Create a new Model Run</h2>
                                <input name="newSosModelRun_name" type="text" onChange={this.handleInputChange}/>
                                <input type="submit" value="Create"/>
                            </form>
                            <input type="button" value="Cancel" onClick={this.closeCreateSosModelRunPopup}/>
                        </div>
                    </Modal>

                    <h2>System-of-Systems Models</h2>
                    <div className="select-container">
                        <select size="5">
                            <option>Energy / Water</option>
                            <option>Population / Solid Waste</option>
                            <option>Transport / Energy 3</option>
                            <option>Solid Waste</option>
                            <option>Energy Demand / Energy Supply</option>
                            <option>Digital Communication</option>
                        </select>
                    </div>
                    <input type="button" value="Edit/View System-of-Systems Configuration" />
                    <input type="button" value="Create a new System-of-Systems Configuration" />

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

                    <input type="button" value="Save Project Configuration" />
                    <input type="button" value="Cancel" />
                </div>
            </div>
        );
    }
}

ProjectOverview.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
};

function mapStateToProps(state) {
    const { sos_model_runs } = state;

    return {
        sos_model_runs: sos_model_runs.items,
        isFetching: sos_model_runs.isFetching
    };
}

export default connect(mapStateToProps)(ProjectOverview);
