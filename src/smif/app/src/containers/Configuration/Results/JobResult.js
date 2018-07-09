import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModelRun } from '../../../actions/actions.js'
import { fetchSosModelRunStatus } from '../../../actions/actions.js'
import { startSosModelRun, killSosModelRun } from '../../../actions/actions.js'

import { saveSosModelRun } from '../../../actions/actions.js'

import Ansi from 'ansi-to-react-with-options'
import Steps, { Step } from 'rc-steps'
import 'rc-steps/assets/index.css'
import 'rc-steps/assets/iconfont.css'
import { CreateButton, DangerButton, SaveButton, ToggleButton } from '../../../components/ConfigForm/General/Buttons'
import stripAnsi from 'strip-ansi'
import moment from 'moment'

import { SosModelRunSummary } from '../../../components/Results/ConfigSummary'

import { FaAngleDoubleUp, FaAngleDoubleDown, FaFloppyO } from 'react-icons/lib/fa'

class SosModelRunConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.modelrun_name = this.props.match.params.name

        this.state = {
            verbosity: 0,
            warm_start: false,
            output_format: 'local_binary',
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

        // Scroll the console output down during running status
        if (this.newData != undefined && this.props.sos_model_run_status.status == 'running' && this.state.followConsole) {
            this.newData.scrollIntoView({ behavior: 'instant' })
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
                verbosity: this.state.verbosity, 
                warm_start: this.state.warm_start,
                output_format: this.state.output_format
            }))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))
    }

    stopJob(modelrun_name) {
        const { dispatch } = this.props
        this.outstanding_request_from = this.props.sos_model_run_status.status
        dispatch(killSosModelRun(modelrun_name))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))
    }

    download(filename, text) {
        var element = document.createElement('a')
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text))
        element.setAttribute('download', filename)
      
        element.style.display = 'none'
        document.body.appendChild(element)
      
        element.click()
      
        document.body.removeChild(element)
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

        var step
        var step_status
        var controls = []

        controls.push(
            <div>
                <div className="form-group row">
                    <label className="col-sm-3 col-form-label">Info messages</label>
                    <div className="col-sm-9 btn-group">
                        <ToggleButton 
                            label1="ON" 
                            label2="OFF" 
                            action1={() => {this.setState({verbosity: 1})}}
                            action2={() => {this.setState({verbosity: 0})}}
                            active1={(this.state.verbosity > 0)} 
                            active2={(this.state.verbosity <= 0)} 
                        />
                    </div>
                    <br/>
                    <label className="col-sm-3 col-form-label">Debug messages</label>
                    <div className="col-sm-9 btn-group">
                        <ToggleButton 
                            label1="ON" 
                            label2="OFF" 
                            action1={() => {this.setState({verbosity: 2})}}
                            action2={() => {this.setState({verbosity: 1})}}
                            active1={(this.state.verbosity > 1)} 
                            active2={(this.state.verbosity <= 1)} 
                        />
                    </div>
                    <br/>
                    <label className="col-sm-3 col-form-label">Warm start</label>
                    <div className="col-sm-9 btn-group">
                        <ToggleButton 
                            label1="ON" 
                            label2="OFF" 
                            action1={() => {this.setState({warm_start: true})}}
                            action2={() => {this.setState({warm_start: false})}}
                            active1={(this.state.warm_start)} 
                            active2={(!this.state.warm_start)} 
                        />
                    </div>
                    <br/>
                    <label className="col-sm-3 col-form-label">Output format</label>
                    <div className="col-sm-9 btn-group">
                        <ToggleButton 
                            label1="Binary" 
                            label2="CSV" 
                            action1={() => {this.setState({output_format: 'local_binary'})}}
                            action2={() => {this.setState({output_format: 'local_csv'})}}
                            active1={(this.state.output_format == 'local_binary')} 
                            active2={(this.state.output_format == 'local_csv')} 
                        />
                    </div>
                </div>
            </div>
        )

        // Reset the outstanding request local property when state changes
        if (this.outstanding_request_from != sos_model_run_status.status) {
            this.outstanding_request_from = null
        }

        switch (sos_model_run_status.status) {
        case 'unknown':
            step = 0
            step_status = 'process'
            controls.push(<CreateButton value='Start Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'queing':
            step = 1
            step_status = 'process'
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'running':
            step = 2
            step_status = 'process'
            controls.push(<DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>)
            this.setRenderInterval(100)
            break
        case 'stopped':
            step = 2
            step_status = 'error'
            controls.push(<SaveButton value='Retry Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'done':
            step = 3
            step_status = 'process'
            controls.push(<CreateButton value='Restart Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        case 'failed':
            step = 2
            step_status = 'error'
            controls.push(<SaveButton value='Retry Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>)
            this.outstanding_request_from == sos_model_run_status.status ? this.setRenderInterval(100) : this.setRenderInterval(0)
            break
        }

        var run_message = null
        switch (sos_model_run_status.status) {
        case 'stopped':
            run_message = 'Modelrun stopped by user'
            break
        case 'failed':
            run_message = 'Modelrun stopped because of error'
            break
        default:
            run_message = 'Modelrun is being executed'
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
                                <div className="row">
                                    <div className="col-10">
                                        {sos_model_run_status.output.split(/\r?\n/).map((status_output, i) =>
                                            <div key={'st_out_line_' + i}><Ansi>{status_output}</Ansi></div>
                                        )}
                                        <div className="cont" ref={(ref) => this.newData = ref}/>
                                    </div>
                                    <div className={'col-2' + ((this.state.followConsole) ? ' align-self-end' : '')}>

                                        <button
                                            type="button"
                                            className="btn btn-outline-dark btn-margin"
                                            onClick={() => {
                                                this.download(moment().format('YMMDD_hmm') + '_' + sos_model_run.name, stripAnsi(sos_model_run_status.output))
                                            }}>
                                            <FaFloppyO/>
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn-outline-dark btn-margin"
                                            onClick={() => {
                                                this.setState({followConsole: !this.state.followConsole})
                                                if ( !this.state.followConsole) {
                                                    this.newData.scrollIntoView({behavior: 'instant'})
                                                } else {
                                                    window.scrollTo(0, 0)
                                                }
                                            }}>
                                            {this.state.followConsole ? (
                                                <FaAngleDoubleUp/>
                                            ) : (
                                                <FaAngleDoubleDown/>
                                            )}
                                        </button>
                                    </div>
                                </div>
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
                                <Steps current={step} status={step_status}>
                                    <Step title="Ready" description="Modelrun is ready to be started" />
                                    <Step title="Queuing" description="Waiting in the queue" />
                                    <Step title="Running" description={run_message} />
                                    <Step title="Completed" description="Modelrun has completed" />
                                </Steps>
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

SosModelRunConfig.propTypes = {
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

export default connect(mapStateToProps)(SosModelRunConfig)
