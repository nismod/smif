import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchScenario } from 'actions/actions.js'
import { fetchDimensions } from 'actions/actions.js'

import { saveScenario } from 'actions/actions.js'

import { setAppFormEdit } from 'actions/actions.js'
import { setAppFormSaveDone } from 'actions/actions.js'
import { setAppNavigate } from 'actions/actions.js'

import ScenarioConfigForm from 'components/ConfigForm/ScenarioConfigForm.js'

class ScenarioConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props

        this.config_name = this.props.match.params.name
    
        dispatch(fetchScenario(this.config_name))
        dispatch(fetchDimensions())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchScenario(this.config_name))
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

    renderScenarioConfig() {
        const { app, scenario, sos_models, dimensions, dispatch } = this.props

        return (
            <div>
                <ScenarioConfigForm
                    scenario={scenario}
                    sos_models={sos_models}
                    dimensions={dimensions}
                    require_provide_full_variant
                    save={app.formReqSave}
                    onNavigate={(dest) => dispatch(setAppNavigate(dest))}
                    onSave={(scenario) => (
                        dispatch(setAppFormSaveDone()),
                        dispatch(saveScenario(scenario))
                    )} 
                    onCancel={() => dispatch(setAppNavigate('/configure/scenarios'))}
                    onEdit={() => dispatch(setAppFormEdit())}/>
            </div>
        )
    }

    render () {
        const {error, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (
            Object.keys(error).includes('SmifDataNotFoundError') ||
            Object.keys(error).includes('SmifValidationError')) {
            return this.renderError(error)
        } else {
            return this.renderScenarioConfig()
        }
    }
}

ScenarioConfig.propTypes = {
    app: PropTypes.object.isRequired,
    scenario: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    dimensions: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {

    return {
        app: state.app,
        scenario: state.scenario.item,
        sos_models: state.sos_models.items,
        dimensions: state.dimensions.items,
        error: ({
            ...state.scenario.error,
            ...state.sos_models.error,
            ...state.dimensions.error
        }),
        isFetching: (
            state.scenario.isFetching ||
            state.sos_models.isFetching ||
            state.dimensions.isFetching
        )
    }
}

export default connect(mapStateToProps)(ScenarioConfig)
