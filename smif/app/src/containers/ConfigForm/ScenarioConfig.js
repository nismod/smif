import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchScenario } from '../../actions/actions.js'
import { fetchSectorModels } from '../../actions/actions.js'
import { fetchScenarioSets } from '../../actions/actions.js'
import { fetchScenarios } from '../../actions/actions.js'
import { fetchNarrativeSets } from '../../actions/actions.js'
import { fetchNarratives } from '../../actions/actions.js'

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
        dispatch(fetchSectorModels())
        dispatch(fetchScenarioSets())
        dispatch(fetchScenarios())
        dispatch(fetchNarrativeSets()) 
        dispatch(fetchNarratives()) 
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

    renderScenarioConfig(scenario, sector_models, scenario_sets, scenarios, narrative_sets, narratives) {
        return (
            <div>
                <h1>Scenario Configuration</h1>         
                <ScenarioConfigForm scenario={scenario} sectorModels={sector_models} scenarioSets={scenario_sets} scenarios={scenarios} narrativeSets={narrative_sets} narratives={narratives} saveScenario={this.saveScenario} cancelScenario={this.returnToPreviousPage}/>            
            </div>
        )
    }

    render () {
        const {scenario, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderScenarioConfig(scenario, sector_models, scenario_sets, scenarios, narrative_sets, narratives)
        }
    }
}

ScenarioConfig.propTypes = {
    scenario: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenario_sets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrative_sets: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {

    return {
        scenario: state.scenario.item,
        sector_models: state.sector_models.items,
        scenario_sets: state.scenario_sets.items,
        scenarios: state.scenarios.items,
        narrative_sets: state.narrative_sets.items,
        narratives: state.narratives.items,
        isFetching: (state.scenario.isFetching)
    }
}

export default connect(mapStateToProps)(ScenarioConfig)