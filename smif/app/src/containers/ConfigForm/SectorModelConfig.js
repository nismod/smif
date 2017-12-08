import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSectorModel } from '../../actions/actions.js'
import { fetchSectorModels } from '../../actions/actions.js'
import { fetchScenarioSets } from '../../actions/actions.js'
import { fetchScenarios } from '../../actions/actions.js'
import { fetchNarrativeSets } from '../../actions/actions.js'
import { fetchNarratives } from '../../actions/actions.js'

import { saveSectorModel } from '../../actions/actions.js'

import SectorModelConfigForm from '../../components/ConfigForm/SectorModelConfigForm.js'

class SectorModelConfig extends Component {
    constructor(props) {
        super(props)

        this.saveSectorModel = this.saveSectorModel.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSectorModel(this.props.match.params.name))
        dispatch(fetchSectorModels())
        dispatch(fetchScenarioSets())
        dispatch(fetchScenarios())
        dispatch(fetchNarrativeSets()) 
        dispatch(fetchNarratives()) 
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveSectorModel(SectorModel) {
        const { dispatch } = this.props
        dispatch(saveSectorModel(SectorModel))
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

    renderSectorModelConfig(sector_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives) {
        return (
            <div>
                <h1>Sector Model Configuration</h1>         
                <SectorModelConfigForm sectorModel={sector_model} sectorModels={sector_models} scenarioSets={scenario_sets} scenarios={scenarios} narrativeSets={narrative_sets} narratives={narratives} saveSectorModel={this.saveSectorModel} cancelSectorModel={this.returnToPreviousPage}/>            
            </div>
        )
    }

    render () {
        const {sector_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderSectorModelConfig(sector_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives)
        }
    }
}

SectorModelConfig.propTypes = {
    sector_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenario_sets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrative_sets: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {
    return {
        sector_model: state.sector_model.item,
        sector_models: state.sector_models.items,
        scenario_sets: state.scenario_sets.items,
        scenarios: state.scenarios.items,
        narrative_sets: state.narrative_sets.items,
        narratives: state.narratives.items,
        isFetching: (state.sos_model.isFetching || state.sos_models.isFetching || state.scenario_sets.isFetching || state.scenarios.isFetching || state.narrative_sets.isFetching || state.narratives.isFetching)
    }
}

export default connect(mapStateToProps)(SectorModelConfig)