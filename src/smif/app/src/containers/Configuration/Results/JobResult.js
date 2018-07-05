import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSosModelRun } from '../../../actions/actions.js'
import { fetchSosModelRunStatus } from '../../../actions/actions.js'
import { startSosModelRun, stopSosModelRun } from '../../../actions/actions.js'

import { saveSosModelRun } from '../../../actions/actions.js'

import Ansi from 'ansi-to-react-with-options'
import Steps, { Step } from 'rc-steps'
import 'rc-steps/assets/index.css'
import 'rc-steps/assets/iconfont.css'
import { CreateButton, DangerButton, SaveButton } from '../../../components/ConfigForm/General/Buttons'

import { SosModelRunSummary } from '../../../components/Results/ConfigSummary'

import { FaAngleDoubleUp, FaAngleDoubleDown } from 'react-icons/lib/fa'

class SosModelRunConfig extends Component {
    constructor(props) {
        super(props)
        this.init = true

        this.saveSosModelRun = this.saveSosModelRun.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)

        this.modelrun_name = this.props.match.params.name
        this.followConsole = false
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
        if (this.newData != undefined && this.props.sos_model_run_status.status == 'running' && this.followConsole) {
            this.newData.scrollIntoView({ behavior: 'instant' })
        }
    }

    setRenderInterval(interval) {
        const { dispatch } = this.props
        if (interval != this.render_interval) {
            this.render_interval = interval
            clearInterval(this.interval)
            this.interval = setInterval(() => dispatch(fetchSosModelRunStatus(this.modelrun_name)), this.render_interval)
        }
    }

    startJob(modelrun_name) {
        const { dispatch } = this.props
        dispatch(startSosModelRun(modelrun_name))
        dispatch(fetchSosModelRunStatus(this.modelrun_name))
    }

    stopJob(modelrun_name) {
        const { dispatch } = this.props
        dispatch(stopSosModelRun(modelrun_name))
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

        var step
        var step_status
        var controls

        switch (sos_model_run_status.status) {
        case 'unknown':
            step = 0
            step_status = 'process'
            controls = <CreateButton value='Start Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>
            this.setRenderInterval(1000)
            break
        case 'queing':
            step = 1
            step_status = 'process'
            controls = <DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>
            this.setRenderInterval(100)
            break
        case 'running':
            step = 2
            step_status = 'process'
            controls = <DangerButton value='Stop Modelrun' onClick={() => {this.stopJob(sos_model_run.name)}}/>
            this.setRenderInterval(100)
            break
        case 'done':
            step = 3
            step_status = 'process'
            controls = <CreateButton value='Restart Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>
            this.setRenderInterval(1000)
            break
        case 'failed':
            step = 2
            step_status = 'error'
            controls = <SaveButton value='Retry Modelrun' onClick={() => {this.startJob(sos_model_run.name)}}/>
            this.setRenderInterval(1000)
            break
        }

        var run_message = sos_model_run_status.status == 'failed' ? 'Modelrun stopped because of error' : 'Modelrun is being executed'

        var console_output = null
        if (sos_model_run_status.status == 'running' || sos_model_run_status.status == 'done' || sos_model_run_status.status == 'failed') {
            console_output = (
                <div className="row">
                    <div className="col-sm">
                        <div className="card">
                            <div className="card-header">
                                Console Output 
                            </div>
                            <div className="card-body">
                                <div className="row">
                                    <div className="col-11">
                                        {sos_model_run_status.output.split(/\r?\n/).map((status_output, i) =>
                                            <div key={'st_out_line_' + i}><Ansi>{status_output}</Ansi></div>
                                        )}
                                        <div className="cont" ref={(ref) => this.newData = ref}/> 
                                    </div>
                                    <div className={'col-1' + ((this.followConsole) ? ' align-self-end' : '')}>
                                        <button
                                            type="button"
                                            className="btn btn-outline-dark"
                                            onClick={() => {
                                                this.followConsole = !this.followConsole 
                                                if ( this.followConsole) {
                                                    this.newData.scrollIntoView({behavior: 'instant'})
                                                } else {
                                                    window.scrollTo(0, 0)
                                                }
                                            }}>
                                            {this.followConsole ? (
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
