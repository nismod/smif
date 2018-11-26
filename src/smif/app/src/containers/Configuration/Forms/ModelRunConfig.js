import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchModelRun } from 'actions/actions.js'
import { fetchSosModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'

import { saveModelRun } from 'actions/actions.js'

import { setAppFormEdit } from 'actions/actions.js'
import { setAppFormSaveDone } from 'actions/actions.js'
import { setAppNavigate } from 'actions/actions.js'

import ModelRunConfigForm from 'components/ConfigForm/ModelRunConfigForm.js'

class ModelRunConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props

        this.config_name = this.props.match.params.name
        
        dispatch(fetchModelRun(this.config_name))
        dispatch(fetchSosModels())
        dispatch(fetchScenarios())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchModelRun(this.config_name))
        }
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderError(error) {
        return (
            <div>
                {            
                    Object.keys(error).map(exception => (
                        <div key={exception} className="alert alert-danger">
                            {exception}
                            {
                                error[exception].map(ex => (
                                    <div key={ex}>
                                        {ex}
                                    </div>
                                ))
                            }
                        </div>
                    ))
                }
            </div>
        )
    }

    renderModelRunConfig() {
        const { app, model_run, sos_models, scenarios, dispatch } = this.props

        return (
            <div key={'sosModel_' + model_run.name}>
                <ModelRunConfigForm 
                    model_run={model_run} 
                    sos_models={sos_models}
                    scenarios={scenarios}
                    save={app.formReqSave}
                    onNavigate={(dest) => dispatch(setAppNavigate(dest))}
                    onSave={(model_run) => (
                        dispatch(setAppFormSaveDone()),
                        dispatch(saveModelRun(model_run))
                    )}
                    onCancel={() => dispatch(setAppNavigate('/configure/model-runs'))}
                    onEdit={() => dispatch(setAppFormEdit())}/>
            </div>
        )
    }

    render () {
        const { error, isFetching } = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (
            Object.keys(error).includes('SmifDataNotFoundError') ||
            Object.keys(error).includes('SmifValidationError')) {
            return this.renderError(error)
        } else {
            return this.renderModelRunConfig()
        }
    }
}

ModelRunConfig.propTypes = {
    app: PropTypes.object.isRequired,
    model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        app: state.app,
        model_run: state.model_run.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        error: ({
            ...state.model_run.error,
            ...state.sos_model.error,
            ...state.scenarios.error
        }),
        isFetching: (
            state.model_run.isFetching || 
            state.sos_models.isFetching ||
            state.scenarios.isFetching
        )
    }
}

export default connect(mapStateToProps)(ModelRunConfig)
