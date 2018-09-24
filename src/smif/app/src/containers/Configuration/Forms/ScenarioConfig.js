import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchScenario } from 'actions/actions.js'
import { saveScenarios } from 'actions/actions.js'
import { createScenario } from 'actions/actions.js'
import { deleteScenario } from 'actions/actions.js'
import { fetchSosModelRuns } from 'actions/actions.js'
import { fetchSosModels } from 'actions/actions.js'

import ScenarioConfigForm from 'components/ConfigForm/ScenarioConfigForm.js'

class ScenarioConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveScenarios = this.saveScenarios.bind(this)
        this.createScenario = this.createScenario.bind(this)
        this.deleteScenario = this.deleteScenario.bind(this)

        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchScenario(this.config_name))
        dispatch(fetchSosModels())
        dispatch(fetchSosModelRuns())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchScenario(this.config_name))
        }
    }

    saveScenarios(Scenario) {
        const { dispatch } = this.props
        dispatch(saveScenarios(Scenario))
        this.returnToPreviousPage()
    }

    deleteScenario(scenario) {
        const { dispatch } = this.props
        dispatch(deleteScenario(scenario))
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/scenarios')
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

    renderScenarioConfig(sos_model_runs, sos_models, scenarios) {
        return (
            <div>
                {/* <ScenarioConfigForm
                    sosModelRuns={sos_model_runs}
                    sosModels={sos_models}
                    scenarios={scenarios}
                    saveScenarioss={this.saveScenarios}
                    cancelScenarios={this.returnToPreviousPage}/> */}
            </div>
        )
    }

    render () {
        const {sos_model_runs, sos_models, scenarios, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderScenarioConfig(sos_model_runs, sos_models, scenarios)
        }
    }
}

ScenarioConfig.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {

    return {
        sos_model_runs: state.sos_model_runs.items,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        isFetching: (
            state.sos_model_runs.isFetching ||
            state.sos_models.isFetching ||
            state.scenarios.isFetching
        )
    }
}

export default connect(mapStateToProps)(ScenarioConfig)
