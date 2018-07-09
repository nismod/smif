import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModelRuns } from 'actions/actions.js'
import ProjectOverviewItem from 'components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'

class JobsOverview extends Component {
    constructor(props) {
        super(props)

        this.param = this.props.match.params.param
    }

    componentDidMount () {
        const { dispatch } = this.props

        this.interval = setInterval(() => dispatch(fetchSosModelRuns(this.param)), 100)
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.param != this.props.match.params.param) {
            this.param = this.props.match.params.param
            dispatch(fetchSosModelRuns(this.param))
        }
    }

    render () {
        const { sos_model_runs, isFetching } = this.props

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
                        <ProjectOverviewItem itemname="SosModelRun" items={sos_model_runs} itemLink="/jobs/runner/" onDelete={this.openDeletePopup} />
                    </div>
                </div>
            </div>
        )
    }
}

JobsOverview.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    const { sos_model_runs } = state

    return {
        sos_model_runs: sos_model_runs.items,
        isFetching: false
    }
}

export default connect(mapStateToProps)(JobsOverview)
