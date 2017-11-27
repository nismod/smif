import React, { Component } from 'react';
import PropTypes from 'prop-types'

class SosModelSelector extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)

    }

    handleChange(event) {
        const target = event.target
        const {onChange} = this.props

        onChange(target.value)
    }

    render() {
        const {sosModelRun, sosModels} = this.props

        let selectedSosModelName = "none"

        if ((sosModelRun && sosModelRun.scenarios) && (sosModels.length > 0)) {
            selectedSosModelName = sosModelRun.sos_model
        }

        return (
            <div className="select-container">
                <select name="sos_model" type="select" value={selectedSosModelName} onChange={(event) => {this.handleChange(event);}}>
                <option disabled="disabled" value="none" >Please select a system-of-systems model</option>
                {
                    sosModels.map((sosModel) => (
                        <option key={sosModel.name} value={sosModel.name}>{sosModel.name}</option>
                    ))
                }
                </select>
            </div>
        )
    }
}

SosModelSelector.propTypes = {
    sosModelRun: PropTypes.object,
    sosModels: PropTypes.array,
    onChange: PropTypes.func
};

export default SosModelSelector;


