import React from 'react';

const SosModelRunConfig = () => (
    <div className="content-wrapper">
        <h2>ModelRun Configuration</h2>

        <div hidden>
            <h3>Loading...</h3>
        </div>

        <div>
            <h3>Error</h3>
        </div>

        <div hidden>
            <h3>General</h3>
            <label>Name:</label>
            <input type="text" name="modelrun_name"  value="unique_model_run_name"/>
            <label>Description:</label>
            <div className="textarea-container">
                <textarea name="textarea" rows="5" value="a description of what the model run contains."/>
            </div>
            <label>Datestamp:</label>
            <input type="datetime-local" value="2017-09-20T12:53:23" disabled="disabled"/>

            <h3>Model</h3>
            <label>System-of-systems model:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Please select a system-of-systems model</option>
                    <option value="energy">energy</option>
                    <option value="water">water</option>
                    <option value="energy-supply-demand">energy-supply-demand</option>
                    <option value="energy-water">energy-water</option>
                </select>
            </div>
            <label>Scenarios:</label>
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

            <label>Narratives:</label>
            <fieldset>
                <legend>Technology</legend>
                <label>
                    <input type="checkbox" />
                    Energy Demand - High Tech
                </label>
                <label>
                    <input type="checkbox" />
                    Solid Waste - High recycling
                </label>
                <label>
                    <input type="checkbox" />
                    Transport - Autonomous driving
                </label>
            </fieldset>
            <fieldset disabled="disabled">
                <legend>Governance</legend>
                <label>
                    <input type="checkbox" />
                    Central Planning
                </label>
                <label>
                    <input type="checkbox" />
                    Hard Brexit
                </label>
                <label>
                    <input type="checkbox" />
                    Soft Brexit
                </label>
            </fieldset>

            <h3>Timesteps</h3>
            <label>Base year:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Please select a base year</option>
                    <option value="2015">2015</option>
                    <option value="2016">2016</option>
                    <option value="2017">2017</option>
                    <option value="2018">2018</option>
                    <option value="2019">2019</option>
                    <option value="2020">2020</option>
                </select>
            </div>
            <label>End year:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Please select an end year</option>
                    <option value="2015">2015</option>
                    <option value="2016">2016</option>
                    <option value="2017">2017</option>
                    <option value="2018">2018</option>
                    <option value="2019">2019</option>
                    <option value="2020">2020</option>
                </select>
            </div>
            <label>Resolution:</label>
            <input type="number" />

            <input type="button" value="Save Model Run Configuration" />
            <input type="button" value="Cancel" />
        </div>
    </div>
);

export default SosModelRunConfig;
