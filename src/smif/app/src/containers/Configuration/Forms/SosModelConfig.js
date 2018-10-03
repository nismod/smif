import React, { Component } from 'react'
import { Redirect } from 'react-router'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModel } from 'actions/actions.js'
import { fetchSectorModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'
import { fetchNarratives } from 'actions/actions.js'

import { saveSosModel } from 'actions/actions.js'
import { acceptSosModel } from 'actions/actions.js'

import SosModelConfigForm from 'components/ConfigForm/SosModelConfigForm.js'

class SosModelConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props
        
        this.saveSosModel = this.saveSosModel.bind(this)
        this.cancelSosModel = this.cancelSosModel.bind(this)
        this.returnToOverview = this.returnToOverview.bind(this)
        
        this.config_name = this.props.match.params.name

        this.state = {
            closeSosmodel: false
        }

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
        this.setState({closeSosmodel: true})
    }

    cancelSosModel() {
        this.setState({closeSosmodel: true})
    }

    returnToOverview() {
        return (
            <Redirect to="/configure/sos-models" />
        )
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderSosModelConfig(sos_model, sector_models, scenarios, narratives, error) {
        return (
            <div key={'sosModel_' + sos_model.name}>
                <SosModelConfigForm sos_model={sos_model} sector_models={sector_models} scenarios={scenarios} narratives={narratives} error={error} saveSosModel={this.saveSosModel} cancelSosModel={this.cancelSosModel} />
            </div>
        )
    }

    render () {
        const {sos_model, sector_models, scenarios, narratives, error, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (this.state.closeSosmodel && Object.keys(error).length == 0) {
            return this.returnToOverview()
        } else {
            return this.renderSosModelConfig(sos_model, sector_models, scenarios, narratives, error)
        }
    }
}

SosModelConfig.propTypes = {
    sos_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
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
        error: state.sos_model.error,
        isFetching: (
            state.sos_model.isFetching || 
            state.sector_models.isFetching || 
            state.scenarios.isFetching || 
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(SosModelConfig)
