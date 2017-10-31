import React from 'react';
import PropTypes from 'prop-types';

const SosModelRunItem = ({ name, description }) => (
    <tr>
        <td>{ name }</td>
        <td>{ description }</td>
    </tr>
);

SosModelRunItem.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired
};

export default SosModelRunItem;
