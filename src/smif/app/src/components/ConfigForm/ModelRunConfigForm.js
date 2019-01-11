import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import { FaPlay } from 'react-icons/fa'
import { PrimaryButton, SecondaryButton } from 'components/ConfigForm/General/Buttons'
import { Range } from 'rc-slider'

class ModelRunConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.cleanModelRun = this.cleanModelRun.bind(this)

        this.state = {
            selected: this.cleanModelRun(this.props.model_run)
        }
        this.form = {
            timestep_base_year: this.props.model_run.timesteps[0],
            timestep_size: (this.props.model_run.timesteps.length <= 1
                ? 1
                : this.props.model_run.timesteps[1] - this.props.model_run.timesteps[0]
            ),
            timestep_number: this.props.model_run.timesteps.length,
        }
    }

    handleChange(key, value) {
        this.props.onEdit()

        if (key == 'scenarios') {
            let newScenarios = Object.assign({}, this.state.selected.scenarios)
            newScenarios[value.scenario] = value.variant

            this.setState({
                selected: update(
                    this.state.selected,
                    {scenarios: {$set: newScenarios}}
                )
            })
        }
        else if (key == 'narratives') {
            let newNarratives = Object.assign({}, this.state.selected.narratives)
            if (newNarratives[value.narrative].includes(value.variant)) {
                newNarratives[value.narrative] = newNarratives[value.narrative].filter(narrative => narrative != value.variant)
            }
            else {
                newNarratives[value.narrative].push(value.variant)
            }

            this.setState({
                selected: update(
                    this.state.selected,
                    {narratives: {$set: newNarratives}}
                )
            })
        }
        else if (key.startsWith('timestep_')) {
            if (key == 'timestep_size') {
                this.form.timestep_size = parseInt(value)
            }
            if (key == 'timestep_base_year') {
                this.form.timestep_base_year = parseInt(value)
            }
            if (key == 'timestep_number') {
                this.form.timestep_number = parseInt(value)
            }

            let timesteps = Array.from(Array(this.form.timestep_number).keys()).map(
                timestep => this.form.timestep_base_year + this.form.timestep_size * timestep
            )

            this.setState({
                selected: update(
                    this.state.selected,
                    {timesteps: {
                        $set: timesteps
                    }}
                )
            })
        }
        else {
            this.setState({
                selected: this.cleanModelRun(
                    update(this.state.selected, {[key]: {$set: value}})
                )
            })
        }
    }

    handleSave() {
        this.props.onSave(this.state.selected)
        this.props.onCancel()
    }

    handleCancel() {
        this.props.onCancel()
    }

    cleanModelRun(sos_model_run) {

        if (sos_model_run.sos_model != '') {
            let scenarios = Object.assign({}, sos_model_run.scenarios)
            let narratives = Object.assign({}, sos_model_run.narratives)

            // Get possible scenarios / narratives from sos_model
            let sos_model = this.props.sos_models.filter(
                sos_model => sos_model.name == sos_model_run.sos_model
            )[0]
            let possible_scenarios = sos_model.scenarios
            let possible_narratives = sos_model.narratives.map(narrative => narrative.name)

            // Make sure each value is present
            possible_scenarios.map(scenario => {
                if (!Object.keys(sos_model_run.scenarios).includes(scenario)) {
                    scenarios[scenario] = ''
                }}
            )
            possible_narratives.map(narrative => {
                if (!Object.keys(sos_model_run.narratives).includes(narrative)) {
                    narratives[narrative] = []
                }}
            )

            // Remove illegal keys
            Object.keys(scenarios).forEach(
                function(scenario_name) {
                    if (!possible_scenarios.includes(scenario_name)) {
                        delete scenarios[scenario_name]
                    }
                }
            )
            Object.keys(narratives).forEach(
                function(narrative_name) {
                    if (!possible_narratives.includes(narrative_name)) {
                        delete narratives[narrative_name]
                    }
                }
            )

            sos_model_run.scenarios = scenarios
            sos_model_run.narratives = narratives
        }
        else {
            sos_model_run.scenarios = {}
            sos_model_run.narratives = {}
        }

        return sos_model_run
    }

    render() {
        const {selected} = this.state

        if (this.props.save) {
            this.props.onSave(this.state.selected)
        }

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-9">
                                <input
                                    className="form-control"
                                    type="text"
                                    disabled="true"
                                    defaultValue={selected.name}
                                    onChange={(event) => this.handleChange('name', event.target.value)}/>
                            </div>
                            <div className="col-sm-1">
                                <button
                                    type="button"
                                    className="btn btn-outline-dark btn-margin"
                                    onClick={() => this.props.onNavigate('/jobs/runner/' + this.state.selected.name)}>
                                    <FaPlay/>
                                </button>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea
                                    className="form-control"
                                    rows="5"
                                    defaultValue={selected.description}
                                    onChange={(event) => this.handleChange('description', event.target.value)}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Created</label>
                            <div className="col-sm-10">
                                <label
                                    className="form-control">
                                    {selected.stamp}
                                </label>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Model</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">System-of-systems model</label>
                            <div className="col-sm-10">
                                <select
                                    className='form-control'
                                    value={this.state.selected.sos_model}
                                    onChange={(event) => this.handleChange('sos_model', event.target.value)}
                                    required>
                                    <option
                                        value=''
                                        disabled>
                                        Please select
                                    </option>
                                    {
                                        this.props.sos_models.map((sos_model) => (
                                            <option
                                                key={sos_model.name}
                                                value={sos_model.name}
                                                title={sos_model.description}>
                                                {sos_model.name}
                                            </option>
                                        ))
                                    }
                                </select>

                                {
                                    this.state.selected.sos_model != '' ?
                                        <PrimaryButton
                                            value="Go to system-of-systems configuration"
                                            onClick={() => this.props.onNavigate('/configure/sos-models/' + this.state.selected.sos_model)}/>
                                        : null
                                }
                            </div>
                        </div>
                        {
                            this.state.selected.sos_model != ''
                                ?
                                <div>
                                    <div className="form-group row"
                                        hidden={
                                            this.props.sos_models.filter(
                                                sos_model => sos_model.name == this.state.selected.sos_model
                                            )[0].scenarios.length == 0
                                                ? true
                                                : false
                                        }>
                                        <label className="col-sm-2 col-form-label">Scenarios</label>
                                        <div className="col-sm-10">
                                            {
                                                this.props.sos_models.filter(
                                                    sos_model => sos_model.name == this.state.selected.sos_model
                                                )[0].scenarios.map(
                                                    sos_model_scenario => (
                                                        <div key={sos_model_scenario} className="card">
                                                            <div className="card-body">
                                                                <h6 className="card-title">{sos_model_scenario}</h6>
                                                                {
                                                                    this.props.scenarios.filter(
                                                                        scenario => scenario.name == sos_model_scenario
                                                                    )[0].variants.map(variant => (
                                                                        <div className="form-check"
                                                                            key={sos_model_scenario + '_' + variant.name}
                                                                            title={variant.description}>
                                                                            <label className="form-check-label">
                                                                                <input
                                                                                    id={'radio_' + sos_model_scenario + '_' + variant.name}
                                                                                    className="form-check-input"
                                                                                    type="radio"
                                                                                    key={variant.name}
                                                                                    value={variant.name}
                                                                                    checked={
                                                                                        this.state.selected.scenarios[sos_model_scenario] == variant.name
                                                                                            ? true : false
                                                                                    }
                                                                                    onChange={(event) => this.handleChange('scenarios', {
                                                                                        scenario: sos_model_scenario,
                                                                                        variant: event.target.value
                                                                                    })} />
                                                                                {variant.name}
                                                                            </label>
                                                                        </div>
                                                                    ))
                                                                }
                                                            </div>
                                                        </div>
                                                    )
                                                )
                                            }
                                        </div>
                                    </div>
                                    <div className="form-group row"
                                        hidden={
                                            this.props.sos_models.filter(
                                                sos_model => sos_model.name == this.state.selected.sos_model
                                            )[0].narratives.length == 0
                                                ? true
                                                : false
                                        }>
                                        <label className="col-sm-2 col-form-label">Narratives</label>
                                        <div className="col-sm-10">
                                            {
                                                this.props.sos_models.filter(
                                                    sos_model => sos_model.name == this.state.selected.sos_model
                                                )[0].narratives.map(
                                                    sos_model_narrative => (
                                                        <div key={sos_model_narrative.name} className="card">
                                                            <div className="card-body">
                                                                <h6 className="card-title">{sos_model_narrative.name}</h6>
                                                                {
                                                                    sos_model_narrative.variants.map(variant => (
                                                                        <div className="form-check"
                                                                            key={sos_model_narrative.name + '_' + variant.name}
                                                                            title={variant.description}>
                                                                            <label className="form-check-label">
                                                                                <input
                                                                                    id={'radio_' + sos_model_narrative.name + '_' + variant.name}
                                                                                    className="form-check-input"
                                                                                    type="checkbox"
                                                                                    name={'narratives'}
                                                                                    key={variant.name}
                                                                                    value={variant.name}
                                                                                    checked={
                                                                                        this.state.selected.narratives[sos_model_narrative.name].includes(variant.name)
                                                                                            ? true : false
                                                                                    }
                                                                                    onChange={() => this.handleChange('narratives', {
                                                                                        narrative: sos_model_narrative.name,
                                                                                        variant: variant.name
                                                                                    })} />
                                                                                {variant.name}
                                                                            </label>
                                                                        </div>
                                                                    ))
                                                                }
                                                            </div>
                                                        </div>
                                                    )
                                                )
                                            }
                                        </div>
                                    </div>
                                </div>
                                : null
                        }
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Timesteps</div>
                    <div className="card-body">

                        <Range
                            min={this.form.timestep_base_year - this.form.timestep_size}
                            max={this.form.timestep_base_year + (this.form.timestep_number * this.form.timestep_size)}
                            marks={
                                this.state.selected.timesteps.reduce(function(acc, cur, i, arr) {
                                    if (arr.length < 10 || (i % parseInt(arr.length / 10) == 0)) {
                                        acc[cur] = cur
                                    }
                                    return acc
                                }, {})
                            }
                            value={this.state.selected.timesteps} />
                        <br/>
                        <br/>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Base year</label>
                            <div className="col-sm-10">
                                <input
                                    className='form-control'
                                    type="number"
                                    min={2000}
                                    max={2100}
                                    value={this.form.timestep_base_year}
                                    onChange={(event) => this.handleChange('timestep_base_year', event.target.value)}
                                />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Step size</label>
                            <div className="col-sm-10">
                                <input
                                    className='form-control'
                                    type="number"
                                    min={1}
                                    max={10}
                                    value={this.form.timestep_size}
                                    onChange={(event) => this.handleChange('timestep_size', event.target.value)}
                                />
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Number of steps</label>
                            <div className="col-sm-10">
                                <input
                                    className='form-control'
                                    type="number"
                                    min={1}
                                    max={100}
                                    value={this.form.timestep_number}
                                    onChange={(event) => this.handleChange('timestep_number', event.target.value)}
                                />
                            </div>
                        </div>

                    </div>
                </div>

                <PrimaryButton value="Save" onClick={this.handleSave}/>
                <SecondaryButton value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ModelRunConfigForm.propTypes = {
    model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    save: PropTypes.bool,
    onNavigate: PropTypes.func,
    onSave: PropTypes.func,
    onCancel: PropTypes.func,
    onEdit: PropTypes.func
}

export default ModelRunConfigForm
