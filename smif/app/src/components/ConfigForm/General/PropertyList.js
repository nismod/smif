import React, { Component } from 'react'
import PropTypes, { string } from 'prop-types'
import update from 'immutability-helper'

import FaTrash from 'react-icons/lib/fa/trash'
import FaPencil from 'react-icons/lib/fa/pencil'

class PropertyList extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)

    }

    handleChange(event) {

        const {name, items, onChange, onDelete} = this.props

        const target = event.currentTarget
        const value = target.type === 'checkbox' ? target.checked : target.value
        const targetname = target.name

        if (targetname == 'edit') {
            onChange(value)
        }
        else if (targetname == 'delete') {
            onDelete(
                {
                    target: {
                        name: name,
                        value: items.splice(value, 1),
                        type: 'array'
                    }
                }
            )
        }
    }

    getColumnSize(columns, editButton, deleteButton) {
        let totalWidth = 100

        editButton ? totalWidth -= 8 : null
        deleteButton ? totalWidth -= 8 : null

        return ((totalWidth / Object.keys(columns).length).toString() + '%')
    }

    getButtonColumn(active) {

        if (active) {
            return (<th width='8%' scope='col'></th>)
        }
        return
    }

    getEditButton(active, itemNumber) {
        if (active) {
            return (
                <td width='8%' >
                    <button type="button" className="btn btn-outline-dark" key={itemNumber} value={itemNumber} name='edit' onClick={this.handleChange}>
                        <FaPencil/>
                    </button>
                </td>
            )
        }
        return
    }

    getDeleteButton(active, itemNumber) {
        if (active) {
            return (
                <td width='8%'>
                    <button type="button" className="btn btn-outline-dark" key={itemNumber} value={itemNumber} name='delete' onClick={this.handleChange}>
                        <FaTrash/>
                    </button>
                </td>
            )
        }
        return
    }

    renderPropertyList(name, items, columns, editButton, deleteButton) {
        //
        return (
            <div>
                <table className="table">

                    <thead className="thead-light">

                        <tr>
                            {
                                Object.keys(columns).map((column, i) => (

                                    <th width={this.getColumnSize(columns, editButton, deleteButton)} key={i} scope="col">
                                        {columns[column]}
                                    </th>
                                ))
                            }
                            {this.getButtonColumn(editButton)}
                            {this.getButtonColumn(deleteButton)}
                        </tr>
                    </thead>


                    <tbody>
                        {
                            Object.keys(items).map((item, i) => (

                                <tr key={i}>
                                    {
                                        Object.keys(columns).map((column, k) => (
                                            <td width={this.getColumnSize(columns, editButton, deleteButton)} key={k}>
                                                {items[item][column]}
                                            </td>

                                        ))
                                    }
                                    {this.getEditButton(editButton, item)}
                                    {this.getDeleteButton(deleteButton, item)}
                                </tr>
                            ))

                        }
                    </tbody>
                </table>
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="property_list_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="property_list_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="property_list_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {itemsName, items, columns, editButton, deleteButton} = this.props

        if (items == null || items == undefined) {
            return this.renderDanger('The items property is not initialised')
        }
        else if (items.length == 0) {
            return this.renderInfo('There are no ' + itemsName + ' configured')
        } else {
            return this.renderPropertyList(itemsName, items, columns, editButton, deleteButton)
        }
    }
}

PropertyList.propTypes = {
    itemsName: PropTypes.string,
    items: PropTypes.array,
    columns: PropTypes.object,
    editButton: PropTypes.bool,
    deleteButton: PropTypes.bool,
    onEdit: PropTypes.func,
    onDelete: PropTypes.func,
}

export default PropertyList
