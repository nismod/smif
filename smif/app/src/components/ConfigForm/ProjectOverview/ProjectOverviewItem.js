import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Link } from 'react-router-dom'

import FaPencil from 'react-icons/lib/fa/pencil'
import FaTrash from 'react-icons/lib/fa/trash'

class SosModelRunItem extends Component {
    constructor(props) {
        super(props)

        this.onDeleteHandler = this.onDeleteHandler.bind(this)
    }

    onDeleteHandler(event) {
        const {onDelete} = this.props

        const target = event.currentTarget
        const name = target.name

        onDelete(
            {
                target: {
                    name: target.value,
                    value: name,
                    type: 'action'
                }
            }
        )
    }

    renderItems(itemname, items, itemLink) {
        return (
            <div>
                <table className="table table-sm fixed">
                    <thead className="thead-light">
                    
                        <tr>
                            <th width="20%" scope="col">Name</th>
                            <th width="66%" scope="col">Description</th>
                            <th width="7%" scope="col"></th>
                            <th width="7%" scope="col"></th>
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
                                            <button type="button" className="btn btn-outline-dark" name={item.name}>
                                                <FaPencil/>
                                            </button>
                                        </Link>
                                    </td> 
                                    <td>
                                        <button type="button" className="btn btn-outline-dark" value={'delete' + itemname} name={item.name} onClick={this.onDeleteHandler}>
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
        const {itemname, items, itemLink} = this.props

        return this.renderItems(itemname, items, itemLink)
    }    
}

SosModelRunItem.propTypes = {
    itemname: PropTypes.string.isRequired,
    items: PropTypes.array.isRequired,
    itemLink: PropTypes.string.isRequired,
    onDelete: PropTypes.func.isRequired
}

export default SosModelRunItem
