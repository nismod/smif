import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchNarrative } from 'actions/actions.js'
import { saveNarrative } from 'actions/actions.js'

import NarrativeConfigForm from 'components/ConfigForm/NarrativeConfigForm.js'

class NarrativeConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveNarrative = this.saveNarrative.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchNarrative(this.config_name))
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchNarrative(this.config_name))
        }
    }

    saveNarrative(Narrative) {
        const { dispatch } = this.props
        dispatch(saveNarrative(Narrative))
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

    renderNarrativeConfig(narrative_set) {
        return (
            <div key={'narrative_set_' + narrative_set.name}>
                <NarrativeConfigForm narrative={narrative_set} saveNarrative={this.saveNarrative} cancelNarrative={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {narrative_set, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderNarrativeConfig(narrative_set)
        }
    }
}

NarrativeConfig.propTypes = {
    narrative_set: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {

    return {
        narrative_set: state.narrative_set.item,
        isFetching: (state.narrative_set.isFetching)
    }
}

export default connect(mapStateToProps)(NarrativeConfig)
