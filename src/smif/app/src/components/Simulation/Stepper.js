import React, { Component } from 'react'
import PropTypes from 'prop-types'

import Steps, {Step} from 'rc-steps'

class Stepper extends Component {

    render() {
        var step
        var step_status

        switch (this.props.status) {
        case 'unstarted':
            step = 0
            step_status = 'process'
            break
        case 'queing':
            step = 1
            step_status = 'process'
            break
        case 'running':
            step = 2
            step_status = 'process'
            break
        case 'stopped':
            step = 2
            step_status = 'error'
            break
        case 'done':
            step = 3
            step_status = 'process'
            break
        case 'failed':
            step = 2
            step_status = 'error'
            break
        }

        var run_message = null
        switch (this.props.status) {
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

        return (
            <div>
                <Steps current={step} status={step_status}>
                    <Step id="step_ready" title="Ready" description="Modelrun is ready to be started" />
                    <Step id="step_queuing" title="Queuing" description="Waiting in the queue" />
                    <Step id="step_running" title="Running" description={run_message} />
                    <Step id="step_completed" title="Completed" description="Modelrun has completed" />
                </Steps>
            </div>
        )
    }
}

Stepper.propTypes = {
    status: PropTypes.string.isRequired,
}

export default Stepper