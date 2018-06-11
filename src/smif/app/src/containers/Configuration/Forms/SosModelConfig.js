import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSosModel } from '../../../actions/actions.js'
import { fetchSectorModels } from '../../../actions/actions.js'
import { fetchScenarioSets } from '../../../actions/actions.js'
import { fetchScenarios } from '../../../actions/actions.js'
import { fetchNarrativeSets } from '../../../actions/actions.js'
import { fetchNarratives } from '../../../actions/actions.js'

import { saveSosModel } from '../../../actions/actions.js'

import SosModelConfigForm from '../../../components/ConfigForm/SosModelConfigForm.js'

class SosModelConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveSosModel = this.saveSosModel.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSosModel(this.config_name))
        dispatch(fetchSectorModels())
        dispatch(fetchScenarioSets())
        dispatch(fetchScenarios())
        dispatch(fetchNarrativeSets())
        dispatch(fetchNarratives())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchSosModel(this.config_name))
        }
    }

    saveSosModel(sosModel) {
        const { dispatch } = this.props
        dispatch(saveSosModel(sosModel))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/sos-models')
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

    renderSosModelConfig(sos_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives) {
        return (
            <div key={'sosModel_' + sos_model.name}>
                <SosModelConfigForm sosModel={sos_model} sectorModels={sector_models} scenarioSets={scenario_sets} scenarios={scenarios} narrativeSets={narrative_sets} narratives={narratives} saveSosModel={this.saveSosModel} cancelSosModel={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sos_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderSosModelConfig(sos_model, sector_models, scenario_sets, scenarios, narrative_sets, narratives)
        }
    }
}

SosModelConfig.propTypes = {
    sos_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenario_sets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrative_sets: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model: state.sos_model.item,
        sector_models: state.sector_models.items,
        scenario_sets: state.scenario_sets.items,
        scenarios: state.scenarios.items,
        narrative_sets: state.narrative_sets.items,
        narratives: state.narratives.items,
        isFetching: (
            state.sos_model.isFetching || 
            state.sos_models.isFetching || 
            state.scenario_sets.isFetching || 
            state.scenarios.isFetching || 
            state.narrative_sets.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(SosModelConfig)
