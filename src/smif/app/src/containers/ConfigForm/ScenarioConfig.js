import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchScenario } from '../../actions/actions.js'
import { fetchScenarioSets } from '../../actions/actions.js'
import { saveScenario } from '../../actions/actions.js'

import ScenarioConfigForm from '../../components/ConfigForm/ScenarioConfigForm.js'

class ScenarioConfig extends Component {
    constructor(props) {
        super(props)

        this.render = this.render.bind(this)

        this.saveScenario = this.saveScenario.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchScenario(this.props.match.params.name))
        dispatch(fetchScenarioSets())
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveScenario(Scenario) {
        const { dispatch } = this.props
        dispatch(saveScenario(Scenario))
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

    renderScenarioConfig(scenario, scenario_sets) {
        return (
            <div>
                <h1>Scenario Configuration</h1>
                <ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} saveScenario={this.saveScenario} cancelScenario={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {scenario, scenario_sets, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (Object.keys(scenario).length === 0 && scenario.constructor === Object) {
            return this.renderLoading()
        } else {
            return this.renderScenarioConfig(scenario, scenario_sets)
        }
    }
}

ScenarioConfig.propTypes = {
    scenario: PropTypes.object.isRequired,
    scenario_sets: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {

    return {
        scenario: state.scenario.item,
        scenario_sets: state.scenario_sets.items,
        isFetching: (state.scenario.isFetching || state.scenario_sets.isFetching)
    }
}

export default connect(mapStateToProps)(ScenarioConfig)
