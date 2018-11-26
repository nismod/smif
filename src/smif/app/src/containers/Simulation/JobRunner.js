import React, { Component } from 'react'
import PropTypes from 'prop-types'
import {connect} from 'react-redux'

import Stepper from 'components/Simulation/Stepper'
import ConsoleDisplay from 'components/Simulation/ConsoleDisplay'
import JobRunControls from 'components/Simulation/JobRunControls'
import {SuccessButton, DangerButton, PrimaryButton} from 'components/ConfigForm/General/Buttons'
import {fetchModelRun, fetchModelRunStatus, startModelRun, killModelRun, saveModelRun} from 'actions/actions.js'
import {ModelRunSummary} from 'components/Simulation/ConfigSummary'


class JobRunner extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveModelRun = this.saveModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.modelrun_name = this.props.match.params.name

        this.state = {
            followConsole: false
        }
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchModelRun(this.modelrun_name))
        dispatch(fetchModelRunStatus(this.modelrun_name))

        this.setRenderInterval(5000)
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.modelrun_name != this.props.match.params.name) {
            this.modelrun_name = this.props.match.params.name
            dispatch(fetchModelRun(this.modelrun_name))
        }
    }

    setRenderInterval(interval) {
        const { dispatch } = this.props
        if (interval != this.render_interval) {
            this.render_interval = interval
            clearInterval(this.interval)
            if (interval > 0) {
                this.interval = setInterval(() => dispatch(fetchModelRunStatus(this.modelrun_name)), this.render_interval)
            }
        }
    }

    startJob(modelrun_name) {
        const { dispatch } = this.props
        this.outstanding_request_from = this.props.model_run_status.status
        dispatch(startModelRun(modelrun_name, 
            {
                verbosity: this.controls.state.verbosity, 
                warm_start: this.controls.state.warm_start,
                output_format: this.controls.state.output_format
            }))
        dispatch(fetchModelRunStatus(this.modelrun_name))
    }

    stopJob(modelrun_name) {
        const { dispatch } = this.props
        this.outstanding_request_from = this.props.model_run_status.status
        dispatch(killModelRun(modelrun_name))
        dispatch(fetchModelRunStatus(this.modelrun_name))
    }

    saveModelRun(ModelRun) {
        const { dispatch } = this.props
        dispatch(saveModelRun(ModelRun))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        this.props.history.push('/configure/sos-model-run')
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

    renderJobRunner(model_run, model_run_status) {

        var controls = []

        controls.push(
            <div>
                <JobRunControls ref={(ref) => this.controls = ref}/>
            </div>
        )

        // Reset the outstanding request local property when state changes
        if (this.outstanding_request_from != model_run_status.status) {
            this.outstanding_request_from = null
        }

        switch (model_run_status.status) {
        case 'unstarted':
            controls.push(<SuccessButton value='Start Modelrun' onClick={() => {this.startJob(model_run.name)}}/>)
            this.outstanding_request_from == model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'queing':
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'running':
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'stopped':
            controls.push(<PrimaryButton value='Retry Modelrun' onClick={() => {this.startJob(model_run.name)}}/>)
            this.outstanding_request_from == model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'done':
            controls.push(<SuccessButton value='Restart Modelrun' onClick={() => {this.startJob(model_run.name)}}/>)
            this.outstanding_request_from == model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'failed':
            controls.push(<PrimaryButton value='Retry Modelrun' onClick={() => {this.startJob(model_run.name)}}/>)
            this.outstanding_request_from == model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        }

        var console_output = null
        if (model_run_status.status == 'running' || model_run_status.status == 'stopped' || model_run_status.status == 'done' || model_run_status.status == 'failed') {
            console_output = (
                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                Console Output
                            </div>
                            <div className="card-body">
                                <ConsoleDisplay name={model_run.name} output={model_run_status.output} status={model_run_status.status}/>
                            </div>
                        </div>
                    </div>
                </div>
            )
        }

        return (
            <div key={'ModelRun_' + model_run.name}>
                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                <Stepper status={model_run_status.status}/>
                            </div>
                            <div className="card-body">
                                <ModelRunSummary ModelRun={model_run} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                Controls
                            </div>
                            <div className="card-body">
                                {controls}
                            </div>
                        </div>
                    </div>
                </div>

                {console_output}
            </div>
        )
    }

    render () {
        const {model_run, model_run_status, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderJobRunner(model_run, model_run_status)
        }
    }
}

JobRunner.propTypes = {
    model_run: PropTypes.object.isRequired,
    model_run_status: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        model_run: state.model_run.item,
        model_run_status: state.model_run_status.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        isFetching: (
            state.model_run.isFetching ||
            state.model_run_status.isFetching ||
            state.sos_models.isFetching ||
            state.scenarios.isFetching
        )
    }
}

export default connect(mapStateToProps)(JobRunner)
