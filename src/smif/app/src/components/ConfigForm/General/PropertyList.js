import React, { Component } from 'react'
import PropTypes, { string } from 'prop-types'
import update from 'immutability-helper'

import FaTrash from 'react-icons/lib/fa/trash'
import FaPencil from 'react-icons/lib/fa/pencil'
import FaExclamationTriangle from 'react-icons/lib/fa/exclamation-triangle'

class PropertyList extends Component {
    constructor(props) {
        super(props)

        this.onEditHandler = this.onEditHandler.bind(this)
        this.onDeleteHandler = this.onDeleteHandler.bind(this)

    }

    onEditHandler(event) {
        const {onEdit} = this.props
        
        const target = event.currentTarget
        const name = target.name

        onEdit(target.value)
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

    getColumnSize(columns, editButton, deleteButton, enableWarnings) {
        let totalWidth = 100

        editButton ? totalWidth -= 8 : null
        deleteButton ? totalWidth -= 8 : null
        enableWarnings ? totalWidth -= 8 : null

        return ((totalWidth / Object.keys(columns).length).toString() + '%')
    }

    getButtonColumn(active) {

        if (active) {
            return (<th width='8%' scope='col'></th>)
        }
        return
    }

    getWarning(active, itemNumber) {
        if (active) {
            if (this.props.rowWarning[itemNumber]) {
                return (
                    <td width='8%'>
                        <FaExclamationTriangle id="warning" />
                    </td>
                )
            } else {
                return (
                    <td width='8%'>
                    </td>
                )
            }
        }
        return
    }

    getEditButton(active, itemNumber) {
        if (active) {
            return (
                <td width='8%' >
                    <button type="button" className="btn btn-outline-dark" key={itemNumber} value={itemNumber} name='edit' onClick={this.onEditHandler}>
                        <FaPencil/>
                    </button>
                </td>
            )
        }
        return
    }

    getDeleteButton(active, itemname, configname) {
        if (active) {
            return (
                <td width='8%'>
                    <button type="button" className="btn btn-outline-dark" key={itemname} value={itemname} name={configname} onClick={this.onDeleteHandler}>
                        <FaTrash/>
                    </button>
                </td>
            )
        }
        return
    }

    renderPropertyList(name, items, columns, editButton, deleteButton, enableWarnings) {
        //
        return (
            <div>
                <table className="table">

                    <thead className="thead-light">

                        <tr>
                            {this.getButtonColumn(enableWarnings)}
                            {
                                Object.keys(columns).map((column, i) => (

                                    <th width={this.getColumnSize(columns, editButton, deleteButton, enableWarnings)} key={i} scope="col">
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

                                <tr id={'property_' + i} key={i}>
                                    {this.getWarning(enableWarnings, item)}
                                    {
                                        Object.keys(columns).map((column, k) => (
                                            <td width={this.getColumnSize(columns, editButton, deleteButton, enableWarnings)} key={k}>
                                                {items[item][column]}
                                            </td>
                                        ))
                                    }
                                    {this.getEditButton(editButton, items[item].name)}
                                    {this.getDeleteButton(deleteButton, name, items[item].name)}
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
        const {itemsName, items, columns, editButton, deleteButton, enableWarnings, rowWarning} = this.props

        if (items == null || items == undefined) {
            return this.renderDanger('The items property is not initialised')
        } else if (enableWarnings == true && rowWarning == undefined) {
            return this.renderInfo('Warnings are enabled but not controlled')
        } else if (items.length == 0) {
            return this.renderInfo('There are no ' + itemsName + ' configured')
        } else {
            return this.renderPropertyList(itemsName, items, columns, editButton, deleteButton, enableWarnings)
        }
    }
}

PropertyList.propTypes = {
    itemsName: PropTypes.string,
    items: PropTypes.array,
    columns: PropTypes.object,
    editButton: PropTypes.bool,
    deleteButton: PropTypes.bool,
    enableWarnings: PropTypes.bool,
    onEdit: PropTypes.func,
    onDelete: PropTypes.func,
    rowWarning: PropTypes.array
}

export default PropertyList
