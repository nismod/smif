import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSosModelRun } from '../../actions/actions.js'
import { fetchSosModels } from '../../actions/actions.js'
import { fetchScenarios } from '../../actions/actions.js'
import { fetchNarratives } from '../../actions/actions.js'

import { saveSosModelRun } from '../../actions/actions.js'

import SosModelRunConfigForm from '../../components/ConfigForm/SosModelRunConfigForm.js'

class SosModelRunConfig extends Component {
    constructor(props) {
        super(props)

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSosModelRun(this.props.match.params.name))
        dispatch(fetchSosModels())
        dispatch(fetchScenarios())
        dispatch(fetchNarratives())
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveSosModelRun(sosModelRun) {
        const { dispatch } = this.props
        dispatch(saveSosModelRun(sosModelRun))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        history.back()
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
            <div>
                <h1>ModelRun Configuration</h1>
                <SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} saveModelRun={this.saveSosModelRun} cancelModelRun={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sos_model_run, sos_models, scenarios, narratives, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
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
        isFetching: (state.sos_model_run.isFetching || state.sos_models.isFetching || state.scenarios.isFetching || state.narratives.isFetching)
    }
}

export default connect(mapStateToProps)(SosModelRunConfig)
