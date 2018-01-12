import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Redirect } from 'react-router-dom'
import { Link } from 'react-router-dom'

import FaPencil from 'react-icons/lib/fa/pencil'
import FaTrash from 'react-icons/lib/fa/trash'

class SosModelRunItem extends Component {
    constructor(props) {
        super(props)

        this.state = {
            redirect: false,
            redirect_to: ''
        }

        this.onEditHandler = this.onEditHandler.bind(this)
        this.onDeleteHandler = this.onDeleteHandler.bind(this)
    }

    onEditHandler(event) {
        const {itemLink} = this.props

        const target = event.currentTarget
        const name = target.id
        
        if (name != undefined) {
            this.setState({
                redirect: true,
                redirect_to: itemLink + name
            })
        }
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
        
        if (this.state.redirect) {
            return <Redirect push to={this.state.redirect_to}/>
        }
        else {
            return (
                <div>
                    <table className="table table-hover table-projectoverview">
                        <thead className="thead-light">
                            <tr>
                                <th className="col-name" scope="col">Name</th>
                                <th className="col-desc" scope="col">Description</th>
                                <th className="col-action" scope="col"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {
                                items.map((item, i) => (
                                    <tr key={i}>
                                        <td id={item.name} onClick={(e) => this.onEditHandler(e)}>
                                            {item.name}
                                        </td>
                                        <td id={item.name} onClick={(e) => this.onEditHandler(e)}>{item.description}</td>
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
