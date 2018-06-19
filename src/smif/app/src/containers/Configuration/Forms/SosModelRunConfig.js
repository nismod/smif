import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSosModelRun } from '../../../actions/actions.js'
import { fetchSosModels } from '../../../actions/actions.js'
import { fetchScenarios } from '../../../actions/actions.js'
import { fetchNarratives } from '../../../actions/actions.js'

import { saveSosModelRun } from '../../../actions/actions.js'

import SosModelRunConfigForm from '../../../components/ConfigForm/SosModelRunConfigForm.js'

class SosModelRunConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.modelrun_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSosModelRun(this.modelrun_name))
        dispatch(fetchSosModels())
        dispatch(fetchScenarios())
        dispatch(fetchNarratives())
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.modelrun_name != this.props.match.params.name) {
            this.modelrun_name = this.props.match.params.name
            dispatch(fetchSosModelRun(this.modelrun_name))
        }
    }

    saveSosModelRun(sosModelRun) {
        const { dispatch } = this.props
        dispatch(saveSosModelRun(sosModelRun))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/sos-model-run')
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderError() {
        return (
            <div className="alert alert-danger">
                Error
            </div>
        )
    }

    renderSosModelConfig(sos_model_run, sos_models, scenarios, narratives) {
        return (
            <div key={'sosModelRun_' + sos_model_run.name}>
                <SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} saveModelRun={this.saveSosModelRun} cancelModelRun={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sos_model_run, sos_models, scenarios, narratives, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderSosModelConfig(sos_model_run, sos_models, scenarios, narratives)
        }
    }
}

SosModelRunConfig.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model_run: state.sos_model_run.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: (
            state.sos_model_run.isFetching || 
            state.sos_models.isFetching || 
            state.scenarios.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(SosModelRunConfig)
