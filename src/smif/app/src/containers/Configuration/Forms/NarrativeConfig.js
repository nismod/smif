import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchNarrative } from 'actions/actions.js'
import { fetchDimensions } from 'actions/actions.js'
import { saveNarrative } from 'actions/actions.js'

import ScenarioNarrativeConfigForm from 'components/ConfigForm/ScenarioNarrativeConfigForm.js'

class NarrativeConfig extends Component {
    constructor(props) {
        super(props)

        this.saveNarrative = this.saveNarrative.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.config_name = this.props.match.params.name
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchNarrative(this.config_name))
        dispatch(fetchDimensions())
    }

    componentDidUpdate() {
        const { dispatch } = this.props

        if (this.config_name != this.props.match.params.name) {
            this.config_name = this.props.match.params.name
            dispatch(fetchNarrative(this.config_name))
        }
    }

    saveNarrative(narrative) {
        const { dispatch } = this.props
        dispatch(saveNarrative(narrative))

        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/narratives')
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderNarrativeConfig(narrative, dimensions) {
        return (
            <div>
                <ScenarioNarrativeConfigForm
                    scenario_narrative={narrative}
                    dimensions={dimensions}
                    saveScenarioNarrative={this.saveNarrative}
                    cancelScenarioNarrative={this.returnToPreviousPage}/>
                    
            </div>
        )
    }

    render () {
        const {narrative, dimensions, isFetching} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderNarrativeConfig(narrative, dimensions)
        }
    }
}

NarrativeConfig.propTypes = {
    narrative: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {

    return {
        narrative: state.narrative.item,
        dimensions: state.dimensions.items,
        isFetching: (
            state.narrative.isFetching ||
            state.dimensions.isFetching
        )
    }
}

export default connect(mapStateToProps)(NarrativeConfig)
