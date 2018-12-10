import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSectorModel } from 'actions/actions.js'
import { fetchSosModels } from 'actions/actions.js'
import { fetchDimensions } from 'actions/actions.js'

import { saveSectorModel } from 'actions/actions.js'

import { setAppFormEdit } from 'actions/actions.js'
import { setAppFormSaveDone } from 'actions/actions.js'
import { setAppNavigate } from 'actions/actions.js'

import SectorModelConfigForm from 'components/ConfigForm/SectorModelConfigForm.js'

class SectorModelConfig extends Component {
    constructor(props) {
        super(props)
        const { dispatch } = this.props
        
        this.config_name = this.props.match.params.name

        dispatch(fetchSectorModel(this.config_name))
        dispatch(fetchSosModels())
        dispatch(fetchDimensions())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchSectorModel(this.config_name))
        }
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

    renderSectorModelConfig() {
        const {app, sector_model, sos_models, dimensions, error, dispatch} = this.props    

        return (
            <div key={sector_model.name}>
                <SectorModelConfigForm 
                    sector_model={sector_model} 
                    sos_models={sos_models} 
                    dimensions={dimensions}
                    error={error}
                    save={app.formReqSave}
                    onNavigate={(dest) => dispatch(setAppNavigate(dest))}
                    onSave={(sector_model) => (
                        dispatch(setAppFormSaveDone()),
                        dispatch(saveSectorModel(sector_model))
                    )} 
                    onCancel={() => dispatch(setAppNavigate('/configure/sector-models'))}
                    onEdit={() => dispatch(setAppFormEdit())}/>
            </div>
        )
    }

    render () {
        const {error, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (
            Object.keys(error).includes('SmifDataNotFoundError') ||
            Object.keys(error).includes('SmifValidationError')) {
            return this.renderError(error)
        } else {
            return this.renderSectorModelConfig()
        }
    }
}

SectorModelConfig.propTypes = {
    app: PropTypes.object.isRequired,
    sector_model: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    dimensions: PropTypes.array.isRequired,
    error: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        app: state.app,
        sector_model: state.sector_model.item,
        sos_models: state.sos_models.items,
        dimensions: state.dimensions.items,
        error: ({
            ...state.sector_model.error,
            ...state.sos_models.error,
            ...state.dimensions.error
        }),
        isFetching: (
            state.sos_models.isFetching ||
            state.sector_model.isFetching ||
            state.dimensions.isFetching
        )
    }
}

export default connect(mapStateToProps)(SectorModelConfig)
