import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Link } from 'react-router-dom';

import { fetchSosModelRun } from '../actions/actions.js';
import SosModelRunConfiguration from '../components/SosModelRunConfiguration.js';

class SosModelRunConfig extends Component {
    componentDidMount() {
        const { dispatch } = this.props;
        dispatch(fetchSosModelRun(this.props.match.params.name));
    }

    render () {
        const {sos_model_run, isFetching} = this.props;
        let config = null;

        if (sos_model_run && sos_model_run.name){
            config = <SosModelRunConfiguration sos_model_run={sos_model_run} />;
        }
        
        return (
            <div>
                <h1>ModelRun Configuration</h1>

                <div hidden={ !isFetching } className="alert alert-primary">
                    Loading...
                </div>

                <div hidden className="alert alert-danger">
                    Error
                </div>

                <div hidden={ isFetching }>            

                    {config}             




                    <label>Scenarios:</label>
                    <fieldset>
                        <legend>Population (ONS)</legend>
                        <label>
                            <input type="radio" name="scenario-population" value="low" />
                            Low
                        </label>
                        <label>
                            <input type="radio" name="scenario-population" value="medium" />
                            Medium
                        </label>
                        <label>
                            <input type="radio" name="scenario-population" value="high" />
                            High
                        </label>
                    </fieldset>

                    <label>Narratives:</label>
                    <fieldset>
                        <legend>Technology</legend>
                        <label>
                            <input type="checkbox" />
                            Energy Demand - High Tech
                        </label>
                        <label>
                            <input type="checkbox" />
                            Solid Waste - High recycling
                        </label>
                        <label>
                            <input type="checkbox" />
                            Transport - Autonomous driving
                        </label>
                    </fieldset>
                    <fieldset disabled="disabled">
                        <legend>Governance</legend>
                        <label>
                            <input type="checkbox" />
                            Central Planning
                        </label>
                        <label>
                            <input type="checkbox" />
                            Hard Brexit
                        </label>
                        <label>
                            <input type="checkbox" />
                            Soft Brexit
                        </label>
                    </fieldset>

                    <h3>Timesteps</h3>
                    <label>Base year:</label>
                    <div className="select-container">
                        <select>
                            <option value="" disabled="disabled" selected="selected">Please select a base year</option>
                            <option value="2015">2015</option>
                            <option value="2016">2016</option>
                            <option value="2017">2017</option>
                            <option value="2018">2018</option>
                            <option value="2019">2019</option>
                            <option value="2020">2020</option>
                        </select>
                    </div>
                    <label>End year:</label>
                    <div className="select-container">
                        <select>
                            <option value="" disabled="disabled" selected="selected">Please select an end year</option>
                            <option value="2015">2015</option>
                            <option value="2016">2016</option>
                            <option value="2017">2017</option>
                            <option value="2018">2018</option>
                            <option value="2019">2019</option>
                            <option value="2020">2020</option>
                        </select>
                    </div>
                    <label>Resolution:</label>
                    <input type="number" />

                    <input type="button" value="Save Model Run Configuration" />
                    <input type="button" value="Cancel" />
                    <Link to="/configure" className="button">
                        Cancel
                    </Link>
                </div>
            </div>
        );
    }
}

SosModelRunConfig.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
};

function mapStateToProps(state) {
    const { sos_model_run } = state;

    return {
        sos_model_run: state.sos_model_run.item,
        isFetching: state.sos_model_run.isFetching
    };
}

export default connect(mapStateToProps)(SosModelRunConfig);
