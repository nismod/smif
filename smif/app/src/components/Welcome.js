import React from 'react';
import '../../static/css/main.css';

const Welcome = () => (
    <div className="content-wrapper">
        <h1>Welcome to Smif!</h1>
        <input type="button" className="primary" value="Start a new project" />
        
        <label>Projectname:</label>
        <input type="text" list="models" name="myModels" />
        <datalist id="models">
            <option value="Project 1"></option>
            <option value="Project 2"></option>
            <option value="Project 3"></option>
            <option value="Project 4"></option>
        </datalist>
        <input type="button" className="primary" value="Go" />
    </div>
);

export default Welcome;
