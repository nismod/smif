import React from 'react';

const ScenarioSelector = ({scenarioSet, scenarios}) => (
    <fieldset>
        <legend>{scenarioSet}</legend>
        {
            scenarios.map((scenario) => (
                <div key={scenario.name}>
                    <input type="radio" key={scenario.name} value={scenario.name}></input>
                    <label>{scenario.name}</label>
                </div>
            ))
        }
    </fieldset>
);

export default ScenarioSelector;