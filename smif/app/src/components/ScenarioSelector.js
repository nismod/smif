import React, { Component } from 'react';
import PropTypes from 'prop-types'

class ScenarioSelector extends Component {
    constructor(props) {
        super(props)

        this.scenarioSelectHandler = this.scenarioSelectHandler.bind(this)
    }

    scenarioSelectHandler(event) {
        const {change_scenario} = this.props

        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        change_scenario(name, value)
    }

    render() {
        const {scenarioSet, scenarios} = this.props

        return (
            <div>
                <fieldset>
                <legend>{scenarioSet}</legend>
                {   
                    scenarios.map((scenario) => (
                        <div key={scenario.name}>
                            <input type="radio" name={scenarioSet} key={scenario.name} value={scenario.name} defaultChecked={scenario.active} onClick={this.scenarioSelectHandler}></input>
                            <label>{scenario.name}</label>
                        </div>
                    ))
                }
                </fieldset>
            </div>
        )
    }
}

ScenarioSelector.propTypes = {
    scenarioSet: PropTypes.string.isRequired,
    scenarios: PropTypes.array.isRequired,
    change_scenario: PropTypes.func.isRequired
};

export default ScenarioSelector;