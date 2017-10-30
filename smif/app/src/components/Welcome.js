import React from 'react';
import { Link } from 'react-router-dom';

const Welcome = () => (
    <div className="jumbotron">
        <h1 className="display-3">Welcome to smif</h1>
        <p className="lead">
            <strong>smif</strong> (a simulation modelling integration framework)
            is designed to support the creation and running of system-of-systems
            models. Aspects of the framework handle inputs and outputs, dependencies
            between models, persistence of data and the communication of state
            between timesteps.
        </p>
        <Link to="/configure" className="btn btn-success btn-lg">
            Set up a system-of-systems project
        </Link>
    </div>
);

export default Welcome;
