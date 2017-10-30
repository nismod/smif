import React from 'react';
import { Link } from 'react-router-dom';

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
            <div className="select-container">
                <select>
                    <option>Modelrun 1</option>
                    <option>Modelrun 2</option>
                    <option>Modelrun 3</option>
                    <option>Modelrun 4</option>
                    <option>Modelrun 5</option>
                    <option>Modelrun 6</option>
                    <option>Modelrun 7</option>
                    <option>Modelrun 8</option>
                    <option>Modelrun 9</option>
                    <option>Modelrun 10</option>
                    <option>Modelrun 11</option>
                    <option>Modelrun 12</option>
                    <option>Modelrun 13</option>
                    <option>Modelrun 14</option>
                    <option>Modelrun 15</option>
                    <option>Modelrun 16</option>
                    <option>Modelrun 17</option>
                    <option>Modelrun 18</option>
                    <option>Modelrun 19</option>
                    <option>Modelrun 20</option>
                </select>
            </div>
            <input type="button" value="Edit/View Model Run" />
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
