import React, { Component } from 'react'

import {ToggleButton} from 'components/ConfigForm/General/Buttons'

class JobRunControls extends Component {

    constructor(props) {
        super(props)

        this.state = {
            verbosity: 0,
            warm_start: false,
            output_format: 'local_binary',
        }
    }

    render() {
        return (
            <div>
                <div className="form-group row">
                    <label className="col-sm-3 col-form-label">Info messages</label>
                    <div className="col-sm-9 btn-group">
                        <ToggleButton
                            id="btn_toggle_info"
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
                            id="btn_toggle_debug"
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
                            id="btn_toggle_warm_start"
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
                            id="btn_toggle_output_format"
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
    }
}

export default JobRunControls
