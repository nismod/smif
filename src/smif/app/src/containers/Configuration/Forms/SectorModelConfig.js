import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSectorModel } from 'actions/actions.js'
import { saveSectorModel } from 'actions/actions.js'
import { fetchSosModels } from 'actions/actions.js'
import { fetchDimensions } from 'actions/actions.js'

import SectorModelConfigForm from 'components/ConfigForm/SectorModelConfigForm.js'

class SectorModelConfig extends Component {
    constructor(props) {
        super(props)

        this.saveSectorModel = this.saveSectorModel.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

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

    saveSectorModel(SectorModel) {
        const { dispatch } = this.props
        dispatch(saveSectorModel(SectorModel))

        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/sector-models')
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderSectorModelConfig(sector_model, sos_models, dimensions) {
        return (
            <div key={sector_model.name}>
                <SectorModelConfigForm sosModels={sos_models} sectorModel={sector_model} dimensions={dimensions} saveSectorModel={this.saveSectorModel} cancelSectorModel={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sector_model, sos_models, dimensions, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderSectorModelConfig(sector_model, sos_models, dimensions)
        }
    }
}

SectorModelConfig.propTypes = {
    sos_models: PropTypes.array.isRequired,
    sector_model: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        sos_models: state.sos_models.items,
        sector_model: state.sector_model.item,
        dimensions: state.dimensions.items,
        isFetching: (
            state.sos_models.isFetching ||
            state.sector_model.isFetching ||
            state.dimensions.isFetching
        )
    }
}

export default connect(mapStateToProps)(SectorModelConfig)
