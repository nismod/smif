import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Link } from 'react-router-dom';

import FaPencil from 'react-icons/lib/fa/pencil';
import FaTrash from 'react-icons/lib/fa/trash';

class SosModelRunItem extends Component {
    constructor(props) {
        super(props)

        this.onDeleteHandler = this.onDeleteHandler.bind(this)
    }

    onDeleteHandler(event) {
        const {onDelete} = this.props

        const target = event.currentTarget
        const name = target.name

        onDelete(name)
    }

    renderItems(items, itemLink) {
        return (
            <div>
                <table className="table table-sm fixed">
                    <col width="20%" />
                    <col width="66%" />
                    <col width="7%" />
                    <col width="7%" />
                    <thead className="thead-light">
                        <tr>
                            <th scope="col">Name</th>
                            <th scope="col">Description</th>
                            <th scope="col"></th>
                            <th scope="col"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            items.map((item, i) => (
                                <tr key={i}>
                                    <td>{item.name}</td>
                                    <td>{item.description}</td>
                                    <td>
                                    <Link to={itemLink + item.name }>
                                        <button name={item.name}>
                                            <FaPencil/>
                                        </button>
                                    </Link>
                                    </td> 
                                    <td>
                                        <button name={item.name} onClick={this.onDeleteHandler}>
                                            <FaTrash/>
                                        </button>
                                    </td> 
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div>
                <font color="red">{message}</font>
            </div>
        )
    }

    render() {
        const {items, itemLink} = this.props

        return this.renderItems(items, itemLink)
    }    
}

SosModelRunItem.propTypes = {
    items: PropTypes.array.isRequired,
    itemLink: PropTypes.string.isRequired,
    onDelete: PropTypes.func.isRequired
};

export default SosModelRunItem;
