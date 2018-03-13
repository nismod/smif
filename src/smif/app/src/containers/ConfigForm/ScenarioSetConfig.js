import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchScenarios } from '../../actions/actions.js'
import { saveScenario } from '../../actions/actions.js'
import { createScenario } from '../../actions/actions.js'
import { deleteScenario } from '../../actions/actions.js'
import { fetchScenarioSet } from '../../actions/actions.js'
import { saveScenarioSet } from '../../actions/actions.js'

import ScenarioSetConfigForm from '../../components/ConfigForm/ScenarioSetConfigForm.js'

class ScenarioSetConfig extends Component {
    constructor(props) {
        super(props)

        this.render = this.render.bind(this)

        this.saveScenarioSet = this.saveScenarioSet.bind(this)
        this.saveScenario = this.saveScenario.bind(this)
        this.createScenario = this.createScenario.bind(this)
        this.deleteScenario = this.deleteScenario.bind(this)

        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchScenarios(this.props.match.params.name))
        dispatch(fetchScenarioSet(this.props.match.params.name))
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveScenarioSet(ScenarioSet) {
        const { dispatch } = this.props
        dispatch(saveScenarioSet(ScenarioSet))
        this.returnToPreviousPage()
    }

    saveScenario(Scenario) {
        const { dispatch } = this.props
        dispatch(saveScenario(Scenario))
    }

    createScenario(scenario) {
        const { dispatch } = this.props
        dispatch(createScenario(scenario['name']))
        scenario['scenario_set'] = this.props.scenario_set['name']
        dispatch(saveScenario(scenario))
        dispatch(fetchScenarios())
    }

    deleteScenario(scenario) {
        const { dispatch } = this.props
        dispatch(deleteScenario(scenario))
        dispatch(fetchScenarios())
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

    renderScenarioSetConfig(scenario_set, scenarios) {
        return (
            <div>
                <h1>Scenario Set Configuration</h1>
                <ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} saveScenarioSet={this.saveScenarioSet} createScenario={this.createScenario} deleteScenario={this.deleteScenario} saveScenario={this.saveScenario} cancelScenarioSet={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {scenarios, scenario_set, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (Object.keys(scenario_set).length === 0 && scenario_set.constructor === Object) {
            return this.renderLoading()
        } else if (Object.keys(scenarios).length === 0 && scenarios.constructor === Object) {
            return this.renderLoading()
        } else {
            return this.renderScenarioSetConfig(scenario_set, scenarios)
        }
    }
}

ScenarioSetConfig.propTypes = {
    scenario_set: PropTypes.object.isRequired,
    scenarios: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {

    return {
        scenario_set: state.scenario_set.item,
        scenarios: state.scenarios.items,
        isFetching: (state.scenario_set.isFetching || state.scenarios.isFetching)
    }
}

export default connect(mapStateToProps)(ScenarioSetConfig)
