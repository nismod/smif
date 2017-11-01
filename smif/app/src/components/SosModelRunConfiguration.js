import React from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

const SosModelRunConfiguration = ({ sos_model_run }) => (
    <div>
        <h3>General</h3>
        <label>Name:</label>
        <input type="text" name="modelrun_name"  value={sos_model_run.name}/>
        <label>Description:</label>
        <div className="textarea-container">
            <textarea name="textarea" rows="5" value={sos_model_run.description}/>
        </div>

        <label>Datestamp:</label>
        <input type="datetime-local" value={sos_model_run.stamp} disabled="disabled"/>

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
    </div>
);

SosModelRunConfiguration.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
};

export default SosModelRunConfiguration;
