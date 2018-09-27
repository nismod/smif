import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchModelRun } from 'actions/actions.js'
import { fetchSosModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'
import { fetchNarratives } from 'actions/actions.js'

import { saveModelRun } from 'actions/actions.js'

import ModelRunConfigForm from 'components/ConfigForm/ModelRunConfigForm.js'

class ModelRunConfig extends Component {
    constructor(props) {
        super(props)

        this.saveModelRun = this.saveModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchModelRun(this.config_name))
        dispatch(fetchSosModels())
        dispatch(fetchScenarios())
        dispatch(fetchNarratives())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchModelRun(this.config_name))
        }
    }

    saveModelRun(sosModel) {
        const { dispatch } = this.props
        dispatch(saveModelRun(sosModel))

        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/model-runs')
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderModelRunConfig() {
        const {model_run, sos_models, scenarios, narratives} = this.props

        return (
            <div key={'sosModel_' + model_run.name}>
                <ModelRunConfigForm 
                    model_run={model_run} 
                    sos_models={sos_models}
                    scenarios={scenarios}
                    narratives={narratives}
                    saveModelRun={this.saveModelRun} 
                    cancelModelRun={this.returnToPreviousPage} />
            </div>
        )
    }

    render () {
        if (this.props.isFetching) {
            return this.renderLoading()
        } else {
            return this.renderModelRunConfig()
        }
    }
}

ModelRunConfig.propTypes = {
    model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        model_run: state.model_run.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: (
            state.model_run.isFetching || 
            state.sos_models.isFetching ||
            state.scenarios.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(ModelRunConfig)
