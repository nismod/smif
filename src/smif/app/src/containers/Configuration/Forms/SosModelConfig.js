import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModel } from 'actions/actions.js'

import { fetchModelRuns } from 'actions/actions.js'
import { fetchSectorModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'

import { saveSosModel } from 'actions/actions.js'

import { setAppFormEdit } from 'actions/actions.js'
import { setAppFormSaveDone } from 'actions/actions.js'
import { setAppNavigate } from 'actions/actions.js'

import SosModelConfigForm from 'components/ConfigForm/SosModelConfigForm.js'

class SosModelConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props

        this.config_name = this.props.match.params.name

        dispatch(fetchSosModel(this.config_name))

        dispatch(fetchModelRuns())
        dispatch(fetchSectorModels())
        dispatch(fetchScenarios())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchSosModel(this.config_name))
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

    renderSosModelConfig() {
        const { app, sos_model, model_runs, sector_models, scenarios, error, dispatch } = this.props

        return (
            <div>
                <SosModelConfigForm 
                    sos_model={sos_model} 
                    model_runs={model_runs} 
                    sector_models={sector_models} 
                    scenarios={scenarios} 
                    error={error}
                    save={app.formReqSave}
                    onNavigate={(dest) => dispatch(setAppNavigate(dest))}
                    onSave={(sos_model) => (
                        dispatch(setAppFormSaveDone()),
                        dispatch(saveSosModel(sos_model))
                    )} 
                    onCancel={() => dispatch(setAppNavigate('/configure/sos-models'))}
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
            return this.renderSosModelConfig()
        }
    }
}

SosModelConfig.propTypes = {
    app: PropTypes.object.isRequired,
    sos_model: PropTypes.object.isRequired,
    model_runs: PropTypes.array.isRequired,
    sector_models: PropTypes.array.isRequired,
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
        sos_model: state.sos_model.item,
        model_runs: state.model_runs.items,
        sector_models: state.sector_models.items,
        scenarios: state.scenarios.items,
        error: ({
            ...state.sos_model.error,
            ...state.sector_models.error,
            ...state.scenarios.error
        }),
        isFetching: (
            state.sos_model.isFetching ||
            state.sector_models.isFetching ||
            state.scenarios.isFetching
        )
    }
}

export default connect(mapStateToProps)(SosModelConfig)
