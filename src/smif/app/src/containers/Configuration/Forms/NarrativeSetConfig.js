import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchNarrativeSet } from '../../../actions/actions.js'
import { saveNarrativeSet } from '../../../actions/actions.js'

import NarrativeSetConfigForm from '../../../components/ConfigForm/NarrativeSetConfigForm.js'

class NarrativeSetConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveNarrativeSet = this.saveNarrativeSet.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchNarrativeSet(this.config_name))
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchNarrativeSet(this.config_name))
        }
    }

    saveNarrativeSet(NarrativeSet) {
        const { dispatch } = this.props
        dispatch(saveNarrativeSet(NarrativeSet))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/narrative-set')
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

    renderNarrativeSetConfig(narrative_set) {
        return (
            <div key={'narrative_set_' + narrative_set.name}>
                <NarrativeSetConfigForm narrativeSet={narrative_set} saveNarrativeSet={this.saveNarrativeSet} cancelNarrativeSet={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {narrative_set, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderNarrativeSetConfig(narrative_set)
        }
    }
}

NarrativeSetConfig.propTypes = {
    narrative_set: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {

    return {
        narrative_set: state.narrative_set.item,
        isFetching: (state.narrative_set.isFetching)
    }
}

export default connect(mapStateToProps)(NarrativeSetConfig)
