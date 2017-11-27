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

    renderSosModelSelector(sosModels, selectedSosModelName) {
        return (
            <div className="select-container">
                <select name="sos_model" type="select" defaultValue={selectedSosModelName} onChange={(event) => {this.handleChange(event);}}>
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

    renderWarning(message) {
        return (
            <div>
                <font color="red">{message}</font>
            </div>
        )
    }

    render() {
        const {sosModelRun, sosModels} = this.props

        if (sosModelRun == null) {
            return this.renderWarning('There is no SosModelRun selected')
        } else if (sosModels == null) {
            return this.renderWarning('There are no SosModels configured')
        } else if (sosModelRun.sos_model == "") {
            return this.renderSosModelSelector(sosModels, "none")
        } else {
            return this.renderSosModelSelector(sosModels, sosModelRun.sos_model)
        }
    }
}

SosModelSelector.propTypes = {
    sosModelRun: PropTypes.object,
    sosModels: PropTypes.array,
    onChange: PropTypes.func
};

export default SosModelSelector;


