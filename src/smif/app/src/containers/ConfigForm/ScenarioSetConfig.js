import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchScenarioSet } from '../../actions/actions.js'
import { saveScenarioSet } from '../../actions/actions.js'

import ScenarioSetConfigForm from '../../components/ConfigForm/ScenarioSetConfigForm.js'

class ScenarioSetConfig extends Component {
    constructor(props) {
        super(props)

        this.render = this.render.bind(this)

        this.saveScenarioSet = this.saveScenarioSet.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

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

    renderScenarioSetConfig(scenario_set) {
        return (
            <div>
                <h1>Scenario Set Configuration</h1>
                <ScenarioSetConfigForm scenarioSet={scenario_set} saveScenarioSet={this.saveScenarioSet} cancelScenarioSet={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {scenario_set, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (Object.keys(scenario_set).length === 0 && scenario_set.constructor === Object) {
            return this.renderLoading()
        } else {
            return this.renderScenarioSetConfig(scenario_set)
        }
    }
}

ScenarioSetConfig.propTypes = {
    scenario_set: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {

    return {
        scenario_set: state.scenario_set.item,
        isFetching: (state.scenario_set.isFetching)
    }
}

export default connect(mapStateToProps)(ScenarioSetConfig)
