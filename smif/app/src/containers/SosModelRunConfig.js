import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSosModelRun } from '../actions/actions.js'
import { fetchSosModels } from '../actions/actions.js'
import { fetchScenarios } from '../actions/actions.js'
import { fetchNarratives } from '../actions/actions.js'

import { resetSosModelRun } from '../actions/actions.js'
import { saveSosModelRun } from '../actions/actions.js'

import SosModelRunConfigForm from '../components/SosModelRunConfigForm.js'

class SosModelRunConfig extends Component {
    componentDidMount() {
        const { dispatch } = this.props
        dispatch(fetchSosModelRun(this.props.match.params.name))
        dispatch(fetchSosModels())
        dispatch(fetchScenarios())
        dispatch(fetchNarratives())

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentWillUnmount() {
        const { dispatch } = this.props
        dispatch(resetSosModelRun())
    }

    saveSosModelRun(sosModelRun) {
        const { dispatch } = this.props
        dispatch(saveSosModelRun(sosModelRun))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        history.back()
    }

    render () {
        const {sos_model_run, sos_models, scenarios, narratives, isFetching} = this.props
        let config_form = null
        let buttons = null;

        if ((sos_model_run && sos_model_run.name) && (sos_models.length > 0) && (narratives.length > 0)){
            
            config_form = <SosModelRunConfigForm 
                sosModelRun={sos_model_run} 
                sosModels={sos_models}
                scenarios={scenarios} 
                narratives={narratives}
                saveModelRun={this.saveSosModelRun}
            />;

            buttons = <div>
                <input type="button" value="Cancel" onClick={this.returnToPreviousPage} /> 
            </div>
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

                    {config_form}            
                    {buttons}

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
