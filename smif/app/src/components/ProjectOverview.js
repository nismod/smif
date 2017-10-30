import React from 'react';
import {Link} from 'react-router';

import '../../static/css/main.css';

const ProjectOverview = () => (

    <div className="content-wrapper" >
        <div>
            <h2>Project Overview</h2>
        </div>

        <div hidden>
            <h3>Loading...</h3>
        </div>

        <div>
            <h3>Error</h3>
        </div>

        <div hidden>
            <label>Projectname:</label>
            <input name="project_name" type="text" value="NISMOD v2.0"/>
            <details>
                <summary>Model Runs</summary>
                <div className="select-container"> 
                    <select size="10">
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
            </details>
            <details>
                <summary>System-of-Systems Models</summary>
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
            </details>
            <details>
                <summary>Simulation Models</summary>
                <div className="select-container">
                    <select size="5">
                        <option>Energy Demand</option>
                        <option>Energy Supply</option>
                        <option>Water</option>
                        <option>Transport</option>
                        <option>Solid Waste</option>
                    </select>
                </div>
                <input type="button" value="Edit/View Simulation Model Configuration" />
                <input type="button" value="Create a new Simulation Model Configuration" />
            </details>
            <details>
                <summary>Scenarios</summary> 
                <input type="button" value="Create a Scenario" />
            </details>
            <details>
                <summary>Narratives</summary> 
                <input type="button" value="Create a Narrative" />
            </details>

            <br/>
            <input type="button" value="Save Project Configuration" />
            <input type="button" value="Cancel" />
        </div>
    </div>
);

export default ProjectOverview;
