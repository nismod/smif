import React, { Component } from 'react'
import { Redirect } from 'react-router'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModel } from 'actions/actions.js'
import { fetchSectorModels } from 'actions/actions.js'
import { fetchScenarios } from 'actions/actions.js'

import { saveSosModel } from 'actions/actions.js'
import { acceptSosModel } from 'actions/actions.js'
import { setAppEditInProgress } from 'actions/actions.js'

import SosModelConfigForm from 'components/ConfigForm/SosModelConfigForm.js'
import { ConfirmPopup } from 'components/ConfigForm/General/Popups.js'

class SosModelConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props

        this.saveForm = this.saveForm.bind(this)
        this.requestCancel = this.requestCancel.bind(this)
        this.confirmCancel = this.confirmCancel.bind(this)
        this.rejectCancel = this.rejectCancel.bind(this)
        this.editForm = this.editForm.bind(this)
        this.returnToOverview = this.returnToOverview.bind(this)

        this.config_name = this.props.match.params.name

        this.state = {
            requestCancel: false,
            cancelForm: false
        }

        dispatch(fetchSosModel(this.config_name))
        dispatch(fetchSectorModels())
        dispatch(fetchScenarios())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            this.setState({cancelForm: false})
            dispatch(fetchSosModel(this.config_name))
        }
    }

    saveForm(sosModel) {
        const { dispatch } = this.props
        dispatch(saveSosModel(sosModel))
        this.setState({cancelForm: true})
    }

    requestCancel() {
        if (this.props.isEdit) {
            this.setState({requestCancel: true})
        } else {
            this.confirmCancel()
        }
    }

    confirmCancel() {
        const { dispatch } = this.props
        dispatch(acceptSosModel())
        this.setState({cancelForm: true})
    }

    rejectCancel() {
        this.setState({requestCancel: false})
    }
    
    editForm() {
        const { dispatch } = this.props
        dispatch(setAppEditInProgress())
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

    renderError(error) {
        return (
            <div>
                {            
                    Object.keys(error).map(exception => (
                        <div key={exception} className="alert alert-danger">
                            {exception}
                            {
                                error[exception].map(ex => (
                                    <div key={ex}>
                                        {ex}
                                    </div>
                                ))
                            }
                        </div>
                    ))
                }
            </div>
        )
    }

    renderSosModelConfig(sos_model, sector_models, scenarios, error) {
        return (
            <div>
                <SosModelConfigForm 
                    sos_model={sos_model} 
                    sector_models={sector_models} 
                    scenarios={scenarios} 
                    error={error} 
                    onSave={this.saveForm} 
                    onCancel={this.requestCancel}
                    onEdit={this.editForm}/>
                <ConfirmPopup 
                    onRequestOpen={this.state.requestCancel}
                    onConfirm={this.confirmCancel}
                    onCancel={this.rejectCancel}/>
            </div>
        )
    }

    render () {
        const {sos_model, sector_models, scenarios, error, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (
            Object.keys(error).includes('SmifDataNotFoundError') ||
            Object.keys(error).includes('SmifValidationError')) {
            return this.renderError(error)
        } else if (this.state.cancelForm && Object.keys(error).length == 0) {
            return this.returnToOverview()
        } else {
            return this.renderSosModelConfig(sos_model, sector_models, scenarios, error)
        }
    }
}

SosModelConfig.propTypes = {
    sos_model: PropTypes.object.isRequired,
    sector_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    isEdit: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model: state.sos_model.item,
        sector_models: state.sector_models.items,
        scenarios: state.scenarios.items,
        error: ({
            ...state.sos_model.error,
            ...state.sector_models.error,
            ...state.scenarios.error
        }),
        isFetching: (
            state.sos_model.isFetching ||
            state.sector_models.isFetching ||
            state.scenarios.isFetching
        ),
        isEdit: state.app.hasPendingChanges
    }
}

export default connect(mapStateToProps)(SosModelConfig)
