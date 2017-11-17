import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Link } from 'react-router-dom';

import { fetchSosModelRun } from '../actions/actions.js';
import { fetchSosModels } from '../actions/actions.js';
import { fetchScenarios } from '../actions/actions.js';
import { fetchNarratives } from '../actions/actions.js';

import { resetSosModelRun } from '../actions/actions.js';

import SosModelRunConfigForm from '../components/SosModelRunConfigForm.js';

class SosModelRunConfig extends Component {
    componentDidMount() {
        const { dispatch } = this.props;
        dispatch(fetchSosModelRun(this.props.match.params.name));
        dispatch(fetchSosModels());        
        dispatch(fetchScenarios());
        dispatch(fetchNarratives());
    }

    componentWillUnmount() {
        const { dispatch } = this.props;
        dispatch(resetSosModelRun());
    }

    render () {
        const {sos_model_run, sos_models, scenarios, narratives, isFetching} = this.props;
        let config = null;

        if ((sos_model_run && sos_model_run.name) && (sos_models.length > 0) && (scenarios.length > 0) && (narratives.length > 0)){
            
            config = <SosModelRunConfigForm 
                sos_model_run={sos_model_run} 
                sos_models={sos_models}
                scenarios={scenarios} 
                narratives={narratives}
            />;
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
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
};

function mapStateToProps(state) {
    return {
        sos_model_run: state.sos_model_run.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: state.sos_model_run.isFetching
    };
}

export default connect(mapStateToProps)(SosModelRunConfig);
