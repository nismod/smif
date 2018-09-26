import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchModelRun } from 'actions/actions.js'

import { saveModelRun } from 'actions/actions.js'

import ModelRunConfigForm from 'components/ConfigForm/ModelRunConfigForm.js'

class ModelRunConfig extends Component {
    constructor(props) {
        super(props)

        this.saveModelRun = this.saveModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchModelRun(this.config_name))
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchModelRun(this.config_name))
        }
    }

    saveModelRun(sosModel) {
        const { dispatch } = this.props
        dispatch(saveModelRun(sosModel))

        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/model-runs')
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderModelRunConfig(model_run) {
        return (
            <div key={'sosModel_' + model_run.name}>
                <ModelRunConfigForm model_run={model_run} saveModelRun={this.saveModelRun} cancelModelRun={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {model_run, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderModelRunConfig(model_run)
        }
    }
}

ModelRunConfig.propTypes = {
    model_run: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        model_run: state.model_run.item,
        isFetching: (
            state.model_run.isFetching || 
            false
        )
    }
}

export default connect(mapStateToProps)(ModelRunConfig)
