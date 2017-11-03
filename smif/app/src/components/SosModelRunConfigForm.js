import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

import ScenarioSelector from '../components/ScenarioSelector.js';

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props);
        this.selectSosModel = this.selectSosModel.bind(this);
        this.pickSosModelByName = this.pickSosModelByName.bind(this);

        console.log(this.props.sos_model_run);
        console.log(this.props.sos_models);
        console.log(this.props.scenarios);

        this.state = {};
        this.state.selectedSosModel = this.pickSosModelByName(this.props.sos_model_run.sos_model);
        this.state.selectedScenarios = this.pickScenariosBySet(this.state.selectedSosModel.scenario_sets);
    }

    pickSosModelByName(name) {
        return this.props.sos_models.filter(
            (sos_model) => sos_model.name === name
        )[0];
    }
    
    pickScenariosBySet(scenario_set) {
        let scenarios_in_sets = new Object();

        for (var i = 0; i < scenario_set.length; i++) {
            scenarios_in_sets[scenario_set[i]] = this.props.scenarios.filter(scenario => scenario.scenario_set === scenario_set[i]);
        };

        return scenarios_in_sets;
    }

    selectSosModel(event) {
        let sos_model = this.pickSosModelByName(event.target.value);
        this.setState({selectedSosModel: sos_model});

        let scenarios = this.pickScenariosBySet(this.state.selectedSosModel.scenario_sets);
        this.setState({selectedScenarios: scenarios});
    }

    render() {
        const { sos_model_run, sos_models, scenarios } = this.props;

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
                <fieldset>            
                    {
                        
                        Object.keys(this.state.selectedScenarios).map((item, i) =>
                            <ScenarioSelector key={i} scenarioSet={item} scenarios={this.state.selectedScenarios[item]} />
                        )
                    }
                </fieldset>
            </div>
        );
    }
}

SosModelRunConfigForm.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired
};

export default SosModelRunConfigForm;
