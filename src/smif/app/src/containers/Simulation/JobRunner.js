import React, { Component } from 'react'
import PropTypes from 'prop-types'
import {connect} from 'react-redux'

import Stepper from 'components/Simulation/Stepper'
import ConsoleDisplay from 'components/Simulation/ConsoleDisplay'
import JobRunControls from 'components/Simulation/JobRunControls'
import {CreateButton, DangerButton, SaveButton} from 'components/ConfigForm/General/Buttons'
import {fetchSosModelRun, fetchSosModelRunStatus, startSosModelRun, killSosModelRun, saveSosModelRun} from 'actions/actions.js'
import {SosModelRunSummary} from 'components/Simulation/ConfigSummary'


class JobRunner extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.modelrun_name = this.props.match.params.name

        this.state = {
            followConsole: false
        }
    }

    componentDidMount() {
        const { dispatch } = this.props

        dispatch(fetchSosModelRun(this.modelrun_name))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))

        this.setRenderInterval(5000)
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    componentDidUpdate() {
        const { dispatch } = this.props
        if (this.modelrun_name != this.props.match.params.name) {
            this.modelrun_name = this.props.match.params.name
            dispatch(fetchSosModelRun(this.modelrun_name))
        }
    }

    setRenderInterval(interval) {
        const { dispatch } = this.props
        if (interval != this.render_interval) {
            this.render_interval = interval
            clearInterval(this.interval)
            if (interval > 0) {
                this.interval = setInterval(() => dispatch(fetchSosModelRunStatus(this.modelrun_name)), this.render_interval)
            }
        }
    }

    startJob(modelrun_name) {
        const { dispatch } = this.props
        this.outstanding_request_from = this.props.sos_model_run_status.status
        dispatch(startSosModelRun(modelrun_name, 
            {
                verbosity: this.controls.state.verbosity, 
                warm_start: this.controls.state.warm_start,
                output_format: this.controls.state.output_format
            }))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))
    }

    stopJob(modelrun_name) {
        const { dispatch } = this.props
        this.outstanding_request_from = this.props.sos_model_run_status.status
        dispatch(killSosModelRun(modelrun_name))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))
    }

    saveSosModelRun(sosModelRun) {
        const { dispatch } = this.props
        dispatch(saveSosModelRun(sosModelRun))
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

    renderSosModelConfig(sos_model_run, sos_model_run_status) {

        var controls = []

        controls.push(
            <div>
                <JobRunControls ref={(ref) => this.controls = ref}/>
            </div>
        )

        // Reset the outstanding request local property when state changes
        if (this.outstanding_request_from != sos_model_run_status.status) {
            this.outstanding_request_from = null
        }

        switch (sos_model_run_status.status) {
        case 'unknown':
            controls.push(<CreateButton value='Start Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'queing':
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'running':
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'stopped':
            controls.push(<SaveButton value='Retry Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'done':
            controls.push(<CreateButton value='Restart Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'failed':
            controls.push(<SaveButton value='Retry Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        }

        var console_output = null
        if (sos_model_run_status.status == 'running' || sos_model_run_status.status == 'stopped' || sos_model_run_status.status == 'done' || sos_model_run_status.status == 'failed') {
            console_output = (
                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                Console Output
                            </div>
                            <div className="card-body">
                                <ConsoleDisplay name={sos_model_run.name} output={sos_model_run_status.output} status={sos_model_run_status.status}/>
                            </div>
                        </div>
                    </div>
                </div>
            )
        }

        return (
            <div key={'sosModelRun_' + sos_model_run.name}>
                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                <Stepper status={sos_model_run_status.status}/>
                            </div>
                            <div className="card-body">
                                <SosModelRunSummary sosModelRun={sos_model_run} />
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
        const {sos_model_run, sos_model_run_status, isFetching} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderSosModelConfig(sos_model_run, sos_model_run_status)
        }
    }
}

JobRunner.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_model_run_status: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired,
    match: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model_run: state.sos_model_run.item,
        sos_model_run_status: state.sos_model_run_status.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: (
            state.sos_model_run.isFetching ||
            state.sos_model_run_status.isFetching ||
            state.sos_models.isFetching ||
            state.scenarios.isFetching ||
            state.narratives.isFetching
        )
    }
}

export default connect(mapStateToProps)(JobRunner)
