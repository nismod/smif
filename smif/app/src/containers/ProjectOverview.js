import React from 'react';
import { Link } from 'react-router-dom';

import SosModelRunItem from '../components/SosModelRunItem.js';

/* mock data for prototyping */
let sos_model_runs = [
    { name: 'Test', description: 'A test description.' },
    { name: 'ew1', description: 'First energy-water run.' }
];

const ProjectOverview = () => (
    <div>
        <h1>Project Overview</h1>

        <div hidden className="alert alert-primary">
            Loading...
        </div>

        <div hidden className="alert alert-danger">
            Error
        </div>

        <div>
            <label>
                Project name
                <input name="project_name" type="text" defaultValue="NISMOD v2.0"/>
            </label>

            <h2>Model Runs</h2>
            <table className="table table-sm">
                <thead className="thead-light">
                    <tr>
                        <th scope="col">Name</th>
                        <th scope="col">Description</th>
                    </tr>
                </thead>
                <tbody>
                    {
                        sos_model_runs.map((sos_model_run) => (
                            <SosModelRunItem key={sos_model_run.name}
                                {...sos_model_run} />
                        ))
                    }
                </tbody>
            </table>
            <input type="button" value="Create a new Model Run" />

            <h2>System-of-Systems Models</h2>
            <div className="select-container">
                <select size="5">
                    <option>Energy / Water</option>
                    <option>Population / Solid Waste</option>
                    <option>Transport / Energy 3</option>
                    <option>Solid Waste</option>
                    <option>Energy Demand / Energy Supply</option>
                    <option>Digital Communication</option>
                </select>
            </div>
            <input type="button" value="Edit/View System-of-Systems Configuration" />
            <input type="button" value="Create a new System-of-Systems Configuration" />

            <h2>Simulation Models</h2>
            <div className="select-container">
                <select>
                    <option>Energy Demand</option>
                    <option>Energy Supply</option>
                    <option>Water</option>
                    <option>Transport</option>
                    <option>Solid Waste</option>
                </select>
            </div>
            <input type="button" value="Edit/View Simulation Model Configuration" />
            <input type="button" value="Create a new Simulation Model Configuration" />

            <h2>Scenarios</h2>
            <input type="button" value="Create a Scenario" />

            <h2>Narratives</h2>
            <input type="button" value="Create a Narrative" />

            <input type="button" value="Save Project Configuration" />
            <input type="button" value="Cancel" />
        </div>
    </div>
);

export default ProjectOverview;
