import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchNarrative } from '../../../actions/actions.js'
import { fetchNarrativeSets } from '../../../actions/actions.js'

import { saveNarrative } from '../../../actions/actions.js'

import NarrativeConfigForm from '../../../components/ConfigForm/NarrativeConfigForm.js'

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
        dispatch(fetchNarrativeSets())
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

    renderNarrativeConfig(narrative, narrative_sets) {
        return (
            <div key={'narrative_name_' + narrative.name}>
                <NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} saveNarrative={this.saveNarrative} cancelNarrative={this.returnToPreviousPage}/>
            </div>
        )
    }

    render () {
        const {narrative, narrative_sets, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
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
        isFetching: (
            state.narrative.isFetching || 
            state.narrative_sets.isFetching
        )
    }
}

export default connect(mapStateToProps)(NarrativeConfig)
