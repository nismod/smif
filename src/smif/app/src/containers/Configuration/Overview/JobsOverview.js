import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModelRuns, startSosModelRun } from '../../../actions/actions.js'


class JobsOverview extends Component {
    constructor(props) {
        super(props)

        this.name = this.props.match.params.name

        this.onStartHandler = this.onStartHandler.bind(this)
    }

    componentDidMount () {
        const { dispatch } = this.props

        this.interval = setInterval(() => dispatch(fetchSosModelRuns(this.name)), 1000);
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.name != this.props.match.params.name) {
            this.name = this.props.match.params.name
            dispatch(fetchSosModelRuns(this.name))
        }
    }

    onStartHandler(event) {
        const { dispatch } = this.props
        const target = event.currentTarget
        const value = target.value

        dispatch(startSosModelRun(value))
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
                    {sos_model_runs.map(sos_model_run =>
                        <div key={'bla' + sos_model_run.name}>
                            <p>{sos_model_run.name}</p>
                            <button
                                type="button"
                                className="btn btn-outline-dark"
                                value={sos_model_run.name}
                                onClick={this.onStartHandler}>
                                Start
                            </button>
                        </div>
                    )}
                </div>
            </div>
        )
    }
}

JobsOverview.propTypes = {
    sos_model_runs: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
}

function mapStateToProps(state) {
    const { sos_model_runs } = state

    return {
        sos_model_runs: sos_model_runs.items,
        isFetching: false
    }
}

export default connect(mapStateToProps)(JobsOverview)
