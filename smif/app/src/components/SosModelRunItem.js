import React from 'react';
import PropTypes from 'prop-types';

import FaPencil from 'react-icons/lib/fa/pencil';
import FaTrash from 'react-icons/lib/fa/trash';

import { Link } from 'react-router-dom';

const SosModelRunItem = ({ name, description }) => (
    <tr>
        <td>{ name }</td>
        <td>{ description }</td>
        <td><Link to={'/configure/sos-model-run/' + name }>Edit</Link></td>
        <td><FaTrash/></td>
    </tr>
);

SosModelRunItem.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired
};

export default SosModelRunItem;
