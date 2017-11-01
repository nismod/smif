import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props);
        this.selectSosModel = this.selectSosModel.bind(this);
        this.pickSosModelByName = this.pickSosModelByName.bind(this);

        this.state = {
            selectedSosModel: this.pickSosModelByName(this.props.sos_model_run.sos_model)
        };
    }

    pickSosModelByName(name) {
        return this.props.sos_models.filter(
            (sos_model) => sos_model.name === name
        )[0];
    }

    selectSosModel(event) {
        let sos_model = this.pickSosModelByName(event.target.value);
        this.setState({selectedSosModel: sos_model});
    }

    render() {
        const { sos_model_run, sos_models } = this.props;

        return (
            <div>
                <h3>General</h3>
                <label>Name:</label>
                <input type="text" name="modelrun_name"  defaultValue={sos_model_run.name}/>
                <label>Description:</label>
                <div className="textarea-container">
                    <textarea name="textarea" rows="5" defaultValue={sos_model_run.description}/>
                </div>

                <label>Datestamp:</label>
                <input type="datetime-local" defaultValue={sos_model_run.stamp} disabled="disabled"/>

                <h3>Model</h3>
                <label>System-of-systems model:</label>
                <div className="select-container">
                    <select value={this.state.selectedSosModel.name} onChange={this.selectSosModel}>
                        <option disabled="disabled" >Please select a system-of-systems model</option>
                        {
                            sos_models.map((sos_model) => (
                                <option key={sos_model.name} value={sos_model.name}>{sos_model.name}</option>
                            ))
                        }
                    </select>
                </div>
                
                <h3>Scenarios</h3>
                {
                    this.state.selectedSosModel.scenario_sets.map((scenario_sets) => (
                        <fieldset key={scenario_sets}>
                            <legend>{scenario_sets}</legend>
                        </fieldset>
                    ))
                }

                <fieldset>
                    <legend>Population (ONS)</legend>
                    <label>
                        <input type="radio" name="scenario-population" value="low" />
                        Low
                    </label>
                    <label>
                        <input type="radio" name="scenario-population" value="medium" />
                        Medium
                    </label>
                    <label>
                        <input type="radio" name="scenario-population" value="high" />
                        High
                    </label>
                </fieldset>
            </div>
        );
    }
}

SosModelRunConfigForm.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired
};

export default SosModelRunConfigForm;
