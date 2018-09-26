import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchScenario } from 'actions/actions.js'
import { fetchDimensions } from 'actions/actions.js'
import { saveScenario } from 'actions/actions.js'

import ScenarioNarrativeConfigForm from 'components/ConfigForm/ScenarioNarrativeConfigForm.js'

class ScenarioConfig extends Component {
    constructor(props) {
        super(props)

        this.saveScenario = this.saveScenario.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

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

    saveScenario(Scenario) {
        const { dispatch } = this.props
        dispatch(saveScenario(Scenario))

        this.returnToPreviousPage()
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

    renderScenarioConfig(scenario, dimensions) {
        return (
            <div>
                <ScenarioNarrativeConfigForm
                    scenario_narrative={scenario}
                    dimensions={dimensions}
                    saveScenario={this.saveScenario}
                    cancelScenario={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {scenario, dimensions, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderScenarioConfig(scenario, dimensions)
        }
    }
}

ScenarioConfig.propTypes = {
    scenario: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {

    return {
        scenario: state.scenario.item,
        dimensions: state.dimensions.items,
        isFetching: (
            state.scenario.isFetching ||
            state.dimensions.isFetching
        )
    }
}

export default connect(mapStateToProps)(ScenarioConfig)
