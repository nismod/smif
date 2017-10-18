import React from 'react';
import FaEdit from 'react-icons/lib/fa/edit';
import FaTrash from 'react-icons/lib/fa/trash';

const SosModelConfig = () => (
    <div className="content-wrapper">
        <h2>System-of-Systems Model Configuration</h2>

        <h3>General</h3>
        <label>Name:</label>
        <input type="text" name="model_name"  value="sos_model_name"/>
        <label>Description:</label>
        <div className="textarea-container">
            <textarea name="textarea" rows="5" value="A system of systems model which encapsulates the future supply and demand of energy for the UK"/>
        </div>

        <h3>Model</h3>
        <fieldset>
            <legend>Scenario Sets</legend>
            <label>
                <input type="checkbox" />
                Population
                <br/>
                <input type="checkbox" />
                Economy
                <br/>
            </label>
        </fieldset>
        <fieldset>
            <legend>Sector Models</legend>
            <label>
                <input type="checkbox" />
                Energy Demand
                <br/>
                <input type="checkbox" />
                Energy Supply
                <br/>
                <input type="checkbox" />
                Transport
                <br/>
                <input type="checkbox" />
                Solid Waste
                <br/>
            </label>
        </fieldset>

        <h3>Dependencies</h3>
        <div className="table-container">
            <table>
                <tr>
                    <th colSpan="2">Source</th>
                    <th colSpan="2">Sink</th>
                    <th width="10px"></th>
                </tr>
                <tr>
                    <th>Model</th>
                    <th>Output</th>
                    <th>Model</th>
                    <th>Input</th>
                    <th></th>
                </tr>
                <tr>
                    <td>population</td>
                    <td>count</td>
                    <td>energy_demand</td>
                    <td>population</td>
                    <td><FaTrash /></td>
                </tr>
                <tr>
                    <td>energy_demand</td>
                    <td>gas_demand</td>
                    <td>energy_supply</td>
                    <td>natural_gas_demand</td>
                    <td><FaTrash /></td>
                </tr>
            </table>
        </div>

        <fieldset>
            <label>Source Model:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a source model</option>
                    <option value="Energy_Demand">Energy Demand</option>
                    <option value="Energy_Supply">Energy Supply</option>
                    <option value="Transport">Transport</option>
                    <option value="Solid_Waste">Solid Waste</option>
                </select>
            </div>
            <label>Source Model Output:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a source model output</option>
                    <option value="population">Population</option>
                    <option value="total_costs">Total costs</option>
                    <option value="fuel_price">Fuel price</option>
                </select>
            </div>
            <label>Sink Model:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a sink model</option>
                    <option value="Energy_Demand">Energy Demand</option>
                    <option value="Energy_Supply">Energy Supply</option>
                    <option value="Transport">Transport</option>
                    <option value="Solid_Waste">Solid Waste</option>
                </select>
            </div>
            <label>Sink Model Input:</label>
            <div className="select-container">
                <select>
                    <option value="" disabled="disabled" selected="selected">Select a sink model input</option>
                    <option value="population">Population</option>
                    <option value="total_costs">Total costs</option>
                    <option value="fuel_price">Fuel price</option>
                </select>
            </div>
            <input type="button" value="Add Dependency" />
        </fieldset>


        <br/>
        <br/>
        <input type="button" value="Save SoS Model Configuration" />
        <input type="button" value="Cancel" />
    </div>
);

export default SosModelConfig;
