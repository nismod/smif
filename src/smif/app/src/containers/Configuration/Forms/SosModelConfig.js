import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModel } from 'actions/actions.js'
import { fetchSectorModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'
import { fetchNarratives } from 'actions/actions.js'

import { saveSosModel } from 'actions/actions.js'

import SosModelConfigForm from 'components/ConfigForm/SosModelConfigForm.js'

class SosModelConfig extends Component {
    constructor(props) {
        super(props)

        this.saveSosModel = this.saveSosModel.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSosModel(this.config_name))
        dispatch(fetchSectorModels())
        dispatch(fetchScenarios())
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

    renderSosModelConfig(sos_model, sector_models, scenarios, narratives) {
        return (
            <div key={'sosModel_' + sos_model.name}>
                <SosModelConfigForm sos_model={sos_model} sector_models={sector_models} scenarios={scenarios} narratives={narratives} saveSosModel={this.saveSosModel} cancelSosModel={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sos_model, sector_models, scenarios, narratives, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderSosModelConfig(sos_model, sector_models, scenarios, narratives)
        }
    }
}

SosModelConfig.propTypes = {
    sos_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model: state.sos_model.item,
        sector_models: state.sector_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: (
            state.sos_model.isFetching || 
            state.sector_models.isFetching || 
            state.scenarios.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(SosModelConfig)
