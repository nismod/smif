import React from 'react';
import { Link } from 'react-router-dom';

const Nav = () => (
    <nav className="nav">
        <Link className="nav-link" to="/" >Home</Link>
        <Link className="nav-link" to="/configure">Configure</Link>
    </nav>
);

export default Nav;
