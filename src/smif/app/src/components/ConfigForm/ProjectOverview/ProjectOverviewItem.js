import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Redirect } from 'react-router-dom'

import {FaTrash, FaPlay} from 'react-icons/lib/fa'

class SosModelRunItem extends Component {
    constructor(props) {
        super(props)

        this.state = {
            redirect: false,
            redirect_to: ''
        }

        this.onEditHandler = this.onEditHandler.bind(this)
        this.onDeleteHandler = this.onDeleteHandler.bind(this)
        this.onStartHandler = this.onStartHandler.bind(this)
    }

    onEditHandler(event) {
        const {itemLink, onClick} = this.props

        const target = event.currentTarget
        const name = target.dataset.name

        if (name != undefined) {
            onClick(itemLink + name)
        }
    }

    onDeleteHandler(event) {
        const {onDelete} = this.props
        const target = event.currentTarget
        const name = target.name

        onDelete({
            target: {
                name: target.value,
                value: name,
                type: 'action'
            }
        })
    }

    onStartHandler(event) {
        const {resultLink, onClick} = this.props
        const target = event.currentTarget
        const name = target.name

        if (name != undefined) {
            onClick(resultLink + name)
        }
    }

<<<<<<< HEAD
    renderItems(itemname, items, itemLink, resultLink, onDelete) {
=======
    renderItems() {
        const {itemname, items, resultLink, onDelete, onClick} = this.props
>>>>>>> Remove unused filters

        if (this.state.redirect) {
            return <Redirect push to={this.state.redirect_to}/>
        }
        else {
            return (
                <table className="table table-hover table-projectoverview">
                    <thead className="thead-light">
                        <tr>
                            <th className="col-name" scope="col">Name</th>
                            <th className="col-desc" scope="col">Description</th>
                            <th hidden={resultLink==undefined} className="col-action" scope="col"></th>
                            <th hidden={onDelete==undefined} className="col-action" scope="col"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            items.map((item, i) => (
                                <tr id={'row_' + item.name} key={i}>
                                    <td
                                        data-name={item.name}
                                        className="col-name"
                                        onClick={(e) => this.onEditHandler(e)}>
                                        {item.name}
                                    </td>
                                    <td
                                        data-name={item.name}
                                        className="col-desc"
                                        onClick={(e) => this.onEditHandler(e)}>
                                        {item.description}
                                    </td>
                                    <td hidden={resultLink==undefined} className="col-action">
                                        <button
                                            id={'btn_start_' + item.name}
                                            type="button"
                                            className="btn btn-outline-dark btn-margin"
                                            value={itemname}
                                            name={item.name}
                                            onClick={this.onStartHandler}>
                                            <FaPlay/>
                                        </button>
                                    </td>
                                    <td hidden={onDelete==undefined} className="col-action">
                                        <button
                                            id={'btn_delete_' + item.name}
                                            type="button"
                                            className="btn btn-outline-dark btn-margin"
                                            value={itemname}
                                            name={item.name}
                                            onClick={this.onDeleteHandler}>
                                            <FaTrash/>
                                        </button>
                                    </td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
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

    renderInfo(message) {
        return (
            <div id="project_overview_item_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {itemname, items, itemLink, resultLink, onDelete} = this.props

        if (itemname == '' || itemname == undefined || itemname == null) {
            return this.renderDanger('There is no itemname configured')
        } else if (itemLink == '' || itemLink == undefined || itemLink == null) {
            return this.renderDanger('There is no itemLink configured')
        } else if (items == null || items == undefined || items.length == 0) {
            return this.renderInfo('There are no items in this list')
        } else {
            return this.renderItems(itemname, items, itemLink, resultLink, onDelete)
        }
    }
}

SosModelRunItem.propTypes = {
    itemname: PropTypes.string,
    items: PropTypes.array,
    itemLink: PropTypes.string,
    resultLink: PropTypes.string,
    onDelete: PropTypes.func,
    onClick: PropTypes.func
}

export default SosModelRunItem
