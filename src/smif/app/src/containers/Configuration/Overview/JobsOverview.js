import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchModelRuns } from 'actions/actions.js'
import { setAppNavigate } from 'actions/actions.js'

import ProjectOverviewItem from 'components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'

import IntroBlock from 'components/ConfigForm/General/IntroBlock.js'

class JobsOverview extends Component {
    constructor(props) {
        super(props)

        this.param = this.props.match.params.param
    }

    componentDidMount () {
        const { dispatch } = this.props

        this.interval = setInterval(() => dispatch(fetchModelRuns(this.param)), 100)
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.param != this.props.match.params.param) {
            this.param = this.props.match.params.param
            dispatch(fetchModelRuns(this.param))
        }
    }

    render () {
        const { model_runs, isFetching } = this.props

        return (
            <div>
                <div hidden={ !isFetching } className="alert alert-primary">
                    Loading...
                </div>

                <div hidden className="alert alert-danger">
                    Error
                </div>

                <div hidden={ isFetching }>
                    <div>
                        <IntroBlock title="Jobs" intro="A job brings together a system-of-systems modelrun configuration and the simulation execution. Each job provides controls to start, stop or restart a modelrun configuration and provides real-time results about its execution. Jobs can be filtered by status by using the navigation pane on the left."/>
                        <ProjectOverviewItem 
                            itemname="ModelRun" 
                            items={model_runs} 
                            itemLink="/jobs/runner/" 
                            onClick={(to) => this.props.dispatch(setAppNavigate(to))} />
                    </div>
                </div>
            </div>
        )
    }
}

JobsOverview.propTypes = {
    model_runs: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    const { model_runs } = state

    return {
        model_runs: model_runs.items,
        isFetching: false
    }
}

export default connect(mapStateToProps)(JobsOverview)
