import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSectorModel } from '../../actions/actions.js'
import { saveSectorModel } from '../../actions/actions.js'
import { fetchSosModels } from '../../actions/actions.js'

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
        dispatch(fetchSosModels())
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

    renderSectorModelConfig(sos_models, sector_model) {
        return (
            <div>
                <h1>Sector Model Configuration</h1>
                <SectorModelConfigForm sosModels={sos_models} sectorModel={sector_model} saveSectorModel={this.saveSectorModel} cancelSectorModel={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {sos_models, sector_model, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderSectorModelConfig(sos_models, sector_model)
        }
    }
}

SectorModelConfig.propTypes = {
    sos_models: PropTypes.array.isRequired,
    sector_model: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {
    return {
        sos_models: state.sos_models.items,
        sector_model: state.sector_model.item,
        isFetching: (state.sector_model.isFetching)
    }
}

export default connect(mapStateToProps)(SectorModelConfig)
