import React from 'react';
import { NavLink } from 'react-router-dom';

const Nav = () => (
    <nav className="nav">
        <NavLink exact className="nav-link" to="/" >Home</NavLink>
        <NavLink exact className="nav-link" to="/configure">Configure</NavLink>
    </nav>
);

export default Nav;
