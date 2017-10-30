import React from 'react';
import { Link } from 'react-router-dom';

const Welcome = () => (
    <div className="content-wrapper">
        <h1>Welcome to Smif!</h1>
        <Link to="/configure">Open Configurator</Link>
    </div>
);

export default Welcome;
