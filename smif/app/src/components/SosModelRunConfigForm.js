import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

import ScenarioSelector from '../components/ScenarioSelector.js';
import NarrativeSelector from '../components/NarrativeSelector.js';

class SosModelRunConfigForm extends Component {
    constructor(props) {
        super(props);
        this.selectSosModel = this.selectSosModel.bind(this);
        this.pickSosModelByName = this.pickSosModelByName.bind(this);

        this.state = {};
        this.state.selectedSosModel = this.pickSosModelByName(this.props.sos_model_run.sos_model);
        this.state.selectedScenarios = this.pickScenariosBySet(this.state.selectedSosModel.scenario_sets);
        this.state.selectedNarratives = this.pickNarrativesBySet(this.state.selectedSosModel.narrative_sets);
    }

    pickSosModelByName(name) {
        return this.props.sos_models.filter(
            (sos_model) => sos_model.name === name
        )[0];
    }
    
    pickScenariosBySet(scenario_set) {
        let scenarios_in_sets = new Object();

        for (var i = 0; i < scenario_set.length; i++) {

            // Get all scenarios that belong to this scenario set
            scenarios_in_sets[scenario_set[i]] = this.props.scenarios.filter(scenario => scenario.scenario_set === scenario_set[i]);

            // Flag the ones that are active in the modelrun configuration
            for (var k = 0; k < scenarios_in_sets[scenario_set[i]].length; k++) {

                scenarios_in_sets[scenario_set[i]][k].active = false;

                this.props.sos_model_run.scenarios.forEach(function(element) {
                    if (scenarios_in_sets[scenario_set[i]][k].name == element[1]) {
                        scenarios_in_sets[scenario_set[i]][k].active = true;
                    }
                });                
            };
        };
        return scenarios_in_sets;
    }

    pickNarrativesBySet(narrative_set) {
        let narratives_in_sets = new Object();

        for (var i = 0; i < narrative_set.length; i++) {

            // Get all narratives that belong to this narrative set
            narratives_in_sets[narrative_set[i]] = this.props.narratives.filter(narrative => narrative.narrative_set === narrative_set[i]);

            // Flag the ones that are active in the modelrun configuration
            for (var k = 0; k < narratives_in_sets[narrative_set[i]].length; k++) {

                narratives_in_sets[narrative_set[i]][k].active = false;
                
                this.props.sos_model_run.narratives.forEach(function(narratives) {
                    narratives[narratives_in_sets[narrative_set[i]][k].narrative_set].forEach(function(narrative) {
                        if (narratives_in_sets[narrative_set[i]][k].name == narrative) {
                            narratives_in_sets[narrative_set[i]][k].active = true;
                        }
                    });
                });
            };                       
        };
        return narratives_in_sets;
    }

    selectSosModel(event) {
        let sos_model = this.pickSosModelByName(event.target.value);
        this.setState({selectedSosModel: sos_model});

        let scenarios = this.pickScenariosBySet(sos_model.scenario_sets);
        this.setState({selectedScenarios: scenarios});

        let narratives = this.pickNarrativesBySet(sos_model.narrative_sets);
        this.setState({selectedNarratives: narratives});
    }

    createBaseyearSelectItems() {
        let years = [];
        for (let i = 2000; i <= 2100; i++) {
            if (i == this.props.sos_model_run.timesteps[0]) {
                years.push(<option key={'baseyear_' + i} selected="selected" value={i}>{i}</option>);
            }
            else {
                years.push(<option key={'baseyear_' + i} value={i}>{i}</option>);
            }
        }
        return years;
    }

    createEndyearSelectItems() {
        let years = [];
        for (let i = 2000; i <= 2100; i++) {
            if (i == this.props.sos_model_run.timesteps[this.props.sos_model_run.timesteps.length - 1]) {
                years.push(<option key={'endyear_' + i} selected="selected" value={i}>{i}</option>);
            }
            else {
                years.push(<option key={'endyear_' + i} value={i}>{i}</option>);
            }
        }
        return years;
    }

    render() {
        const { sos_model_run, sos_models, scenarios, narratives } = this.props;

        console.log(sos_model_run);

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
                <input type="datetime-local" defaultValue=""/>

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

                <h3>Narratives</h3>
                <fieldset>            
                    {         
                        Object.keys(this.state.selectedNarratives).map((item, i) =>
                            <NarrativeSelector key={i} narrativeSet={item} narratives={this.state.selectedNarratives[item]} />
                        )
                    }
                </fieldset>

                <h3>Timesteps</h3>
                <label>Base year:</label>
                <div className="select-container">
                    <select>
                        <option value="" disabled="disabled">Please select a base year</option>
                        {this.createBaseyearSelectItems()}
                    </select>
                </div>
                <label>End year:</label>
                <div className="select-container">
                    <select>
                        <option value="" disabled="disabled">Please select an end year</option>
                        {this.createEndyearSelectItems()}
                    </select>
                </div>
                <label>Resolution:</label>
            </div>
        );
    }
}

SosModelRunConfigForm.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired
};

export default SosModelRunConfigForm;
