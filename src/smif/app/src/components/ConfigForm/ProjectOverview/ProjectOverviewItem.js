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
                                    <tr id={"row_" + item.name} key={i}>
                                        <td id={item.name} onClick={(e) => this.onEditHandler(e)}>
                                            {item.name}
                                        </td>
                                        <td id={item.name} onClick={(e) => this.onEditHandler(e)}>{item.description}</td>
                                        <td>
                                            <button type="button" className="btn btn-outline-dark" value={itemname} name={item.name} onClick={this.onDeleteHandler}>
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

    renderDanger(message) {
        return (
            <div id="project_overview_item_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="project_overview_item_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="project_overview_item_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {itemname, items, itemLink} = this.props

        if (itemname == "" || itemname == undefined || itemname == null) {
            return this.renderDanger('There is no itemname configured')
        } else if (itemLink == "" || itemLink == undefined || itemLink == null) {
            return this.renderDanger('There is no itemLink configured')
        } else if (items == null || items == undefined || items.length == 0) {
            return this.renderInfo('There are no items in this list')
        } else {
            return this.renderItems(itemname, items, itemLink)
        }
    }
}

SosModelRunItem.propTypes = {
    itemname: PropTypes.string,
    items: PropTypes.array,
    itemLink: PropTypes.string,
    onDelete: PropTypes.func
}

export default SosModelRunItem
