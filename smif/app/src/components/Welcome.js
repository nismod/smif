import React from 'react';
import { Link } from 'react-router';

import '../../static/css/main.css';

const Welcome = () => (
    <div className="content-wrapper">
        <div>
            <h1>Welcome to Smif!</h1>
            <li><Link to="/configure" activeClassName="active">Open Configurator</Link></li>
        </div>
    </div>
);

export default Welcome;