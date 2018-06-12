import React from 'react';

const Welcome = () => (
    <div className="jumbotron jumbotron-fluid">
        <h1>Welcome to smif</h1>
        <p>
            <strong>smif</strong> (a simulation modelling integration framework)
            is designed to support the creation and running of system-of-systems
            models. Aspects of the framework handle inputs and outputs, dependencies
            between models, persistence of data and the communication of state
            between timesteps.
        </p>
    </div>
);

export default Welcome;
