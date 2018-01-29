import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchNarrative } from '../../actions/actions.js'
import { fetchNarrativeSets } from '../../actions/actions.js'

import { saveNarrative } from '../../actions/actions.js'

import NarrativeConfigForm from '../../components/ConfigForm/NarrativeConfigForm.js'

class NarrativeConfig extends Component {
    constructor(props) {
        super(props)

        this.saveNarrative = this.saveNarrative.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchNarrative(this.props.match.params.name))
        dispatch(fetchNarrativeSets())
    }

    componentWillReceiveProps() {
        this.forceUpdate()
    }

    saveNarrative(Narrative) {
        const { dispatch } = this.props
        dispatch(saveNarrative(Narrative))
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

    renderNarrativeConfig(narrative, narrative_sets) {
        return (
            <div>
                <h1>Narrative Configuration</h1>
                <NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} saveNarrative={this.saveNarrative} cancelNarrative={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {narrative, narrative_sets, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else if (Object.keys(narrative).length === 0 && narrative.constructor === Object) {
            return this.renderLoading()
        } else {
            return this.renderNarrativeConfig(narrative, narrative_sets)
        }
    }
}

NarrativeConfig.propTypes = {
    narrative: PropTypes.object.isRequired,
    narrative_sets: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired
}

function mapStateToProps(state) {
    return {
        narrative: state.narrative.item,
        narrative_sets: state.narrative_sets.items,
        isFetching: (state.narrative.isFetching || state.narrative_sets.isFetching)
    }
}

export default connect(mapStateToProps)(NarrativeConfig)
