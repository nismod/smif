import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchNarrativeSet } from '../../actions/actions.js'
import { saveNarrativeSet } from '../../actions/actions.js'

import NarrativeSetConfigForm from '../../components/ConfigForm/NarrativeSetConfigForm.js'

class NarrativeSetConfig extends Component {
    constructor(props) {
        super(props)

        this.render = this.render.bind(this)

        this.saveNarrativeSet = this.saveNarrativeSet.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchNarrativeSet(this.props.match.params.name))
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveNarrativeSet(NarrativeSet) {
        const { dispatch } = this.props
        dispatch(saveNarrativeSet(NarrativeSet))
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

    renderNarrativeSetConfig(narrative_set) {
        return (
            <div>
                <h1>Narrative Set Configuration</h1>
                <NarrativeSetConfigForm narrativeSet={narrative_set} saveNarrativeSet={this.saveNarrativeSet} cancelNarrativeSet={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {narrative_set, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (Object.keys(narrative_set).length === 0 && narrative_set.constructor === Object) {
            return this.renderLoading()
        } else {
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
