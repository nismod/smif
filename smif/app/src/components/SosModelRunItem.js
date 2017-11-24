import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

import FaPencil from 'react-icons/lib/fa/pencil';
import FaTrash from 'react-icons/lib/fa/trash';

class SosModelRunItem extends Component {
    constructor(props) {
        super(props)

        this.onEditHandler = this.onEditHandler.bind(this)
        this.onDeleteHandler = this.onDeleteHandler.bind(this)
    }

    onEditHandler(event) {
        const {onEdit} = this.props
        
        const target = event.currentTarget
        const name = target.name

        onEdit(name)        
    }

    onDeleteHandler(event) {
        const {onDelete} = this.props

        const target = event.currentTarget
        const name = target.name

        onDelete(name)
    }

    render() {
        const {name, description} = this.props
        //<FaPencil name="Pencil"/>
        return (
            <tr>
                <td>{name}</td>
                <td>{description}</td>
                <td>
                    <Link to={'/configure/sos-model-run/' + name }>
                        <button name={name}>
                            <FaPencil/>
                        </button>
                    </Link>
                </td> 
                <td>
                    <button name={name} onClick={this.onDeleteHandler}>
                        <FaTrash/>
                    </button>
                </td> 
            </tr>
        )
    }    
}

SosModelRunItem.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    onDelete: PropTypes.func.isRequired
};

export default SosModelRunItem;
