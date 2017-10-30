import React from 'react';

import FaTrash from 'react-icons/lib/fa/trash';

import '../../static/css/main.css';

const SimulationModelConfig = () => (
    <div className="content-wrapper">
        <h2>Simulation Model Configuration</h2>

        <h3>General</h3>
        <label>Name:</label>
        <input type="text" name="sector_model_name"  value="energy_demand"/>
        <label>Description:</label>
        <div className="textarea-container">
            <textarea name="textarea" rows="5" value="Computes the energy demand of the UK population for each timestep"/>
        </div>

        <h3>Model</h3>
        <label>Wrapper Location:</label>
        <input type="text" />
        <label>Class Name:</label>
        <input type="text" />

        <h3>Inputs</h3>
        <div className="table-container">
            <table>
                <tr>
                    <th>Name</th>
                    <th>Spatial Resolution</th>
                    <th>Temporal Resolution</th>
                    <th>Units</th>
                    <th width="10px"></th>
                </tr>
                <tr>
                    <td>Energy Demand per household</td>
                    <td>Household</td>
                    <td>Daily</td>
                    <td>KWh</td>
                    <td><FaTrash /></td>
                </tr>
                <tr>
                    <td>Energy Demand per district</td>
                    <td>LAD</td>
                    <td>Montly</td>
                    <td>MWh</td>
                    <td><FaTrash /></td>
                </tr>
                <tr>
                    <td>Total national Energy Demand</td>
                    <td>National</td>
                    <td>Yearly</td>
                    <td>GWh</td>
                    <td><FaTrash /></td>
                </tr>
            </table>
        </div>
        <fieldset>
            <label>Name:</label>
                <input type="text" />
            <label>Spatial Resolution:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a spatial resolution</option>
                    <option value="Household">Household</option>
                    <option value="LAD">LAD</option>
                    <option value="National">National</option>
                </select>
            </div>
            <label>Temporal Resolution:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a temporal resolution</option>
                    <option value="Hourly">Hourly</option>
                    <option value="Daily">Daily</option>
                    <option value="Weekly">Weekly</option>
                    <option value="Monthly">Monthly</option>
                    <option value="Yearly">Yearly</option>
                </select>
            </div>
            <label>Units:</label>
            <input type="text" list="units" name="myUnits" />
            <datalist id="units">
                <option value="people">people</option>
                <option value="KWh">KWh</option>
                <option value="MWh">MWh</option>
                <option value="GWh">GWh</option>
                <option value="pounds">pounds</option>
            </datalist>
            <input type="button" value="Add Input" />
        </fieldset>

        <h3>Outputs</h3>
        <div className="table-container">
            <table>
                <tr>
                    <th>Name</th>
                    <th>Spatial Resolution</th>
                    <th>Temporal Resolution</th>
                    <th>Units</th>
                    <th width="10px"></th>
                </tr>
                <tr>
                    <td>Energy Demand per household</td>
                    <td>Household</td>
                    <td>Daily</td>
                    <td>KWh</td>
                    <td><FaTrash /></td>
                </tr>
                <tr>
                    <td>Energy Demand per district</td>
                    <td>LAD</td>
                    <td>Montly</td>
                    <td>MWh</td>
                    <td><FaTrash /></td>
                </tr>
                <tr>
                    <td>Total national Energy Demand</td>
                    <td>National</td>
                    <td>Yearly</td>
                    <td>GWh</td>
                    <td><FaTrash /></td>
                </tr>
            </table>
        </div>
        <fieldset>
            <label>Name:</label>
                <input type="text" />
            <label>Spatial Resolution:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a spatial resolution</option>
                    <option value="Household">Household</option>
                    <option value="LAD">LAD</option>
                    <option value="National">National</option>
                </select>
            </div>
            <label>Temporal Resolution:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a temporal resolution</option>
                    <option value="Hourly">Hourly</option>
                    <option value="Daily">Daily</option>
                    <option value="Weekly">Weekly</option>
                    <option value="Monthly">Monthly</option>
                    <option value="Yearly">Yearly</option>
                </select>
            </div>
            <label>Units:</label>
            <input type="text" list="units" name="myUnits" />
            <datalist id="units">
                <option value="people">people</option>
                <option value="KWh">KWh</option>
                <option value="MWh">MWh</option>
                <option value="GWh">GWh</option>
                <option value="pounds">pounds</option>
            </datalist>
            <input type="button" value="Add Output" />
        </fieldset>

        <h3>Parameters</h3>
        <div className="table-container">
            <table>
                <tr>
                    <th></th>
                    <th colSpan="2">Absolute Range</th>
                    <th colSpan="2">Suggested Range</th>
                    <th></th>
                    <th></th>
                    <th width="10px"></th>
                </tr>
                <tr>
                    <th>Name</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Default</th>
                    <th>Units</th>
                    <th></th>
                </tr>
                <tr>
                    <td><abbr title="Difference in floor area per person in end year compared to base year">assump_diff_floorarea_pp</abbr></td>
                    <td>0.5</td>
                    <td>2</td>
                    <td>0.5</td>
                    <td>2</td>
                    <td>1</td>
                    <td>percentage</td>
                    <td><FaTrash /></td>
                </tr>
            </table>
        </div>
        <fieldset>
            <label>Name:</label>
            <input type="text" />
            <label>Description:</label>
            <input type="text" />
            <label>Absolute Range:</label>
            <fieldset>
                <i>Minimal</i>
                <input type="number" />
                <i>Maximum</i>
                <input type="number" />
            </fieldset>
            <label>Suggested Range:</label>
            <fieldset>
                <i>Minimal</i>
                <input type="number" />
                <i>Maximum</i>
                <input type="number" />
                </fieldset>
            <label>Default:</label>
            <input type="number" />
            <label>Units:</label>
            <input type="text" list="units" name="myUnits" />
            <datalist id="units">
                <option value="people">people</option>
                <option value="KWh">KWh</option>
                <option value="MWh">MWh</option>
                <option value="GWh">GWh</option>
                <option value="pounds">pounds</option>
            </datalist>
            <input type="button" value="Add Parameter" />
        </fieldset>

        <h3>Interventions</h3>
        <div className="table-container">
            <table>
                <tr>
                    <th>Intervention Files</th>
                    <th width="10px"></th>
                </tr>
                <tr>
                    <td>energy_demand.yml</td>
                    <td><FaTrash /></td>
                </tr>
            </table>
        </div>
        <fieldset>
            <label>Filename:</label>
                <input type="text" />
            <input type="button" value="Add Intervention" />
        </fieldset>

        <h3>Initial Conditions</h3>
        <label>Filename:</label>
        <input type="text" />

        <br/>
        <br/>
        <input type="button" value="Save Model Configuration" />
        <input type="button" value="Cancel" />
    </div>
);

export default SimulationModelConfig;
