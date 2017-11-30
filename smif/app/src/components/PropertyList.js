import React, { Component } from 'react'
import PropTypes, { string } from 'prop-types'
import update from 'react-addons-update'

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

    getColumnSize(items, editButton, deleteButton) {
        let totalWidth = 100

        editButton ? totalWidth -= 8 : null
        deleteButton ? totalWidth -= 8 : null

        return ((totalWidth / items.length).toString() + '%')
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
                <td>
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
                <td>
                    <button type="button" className="btn btn-outline-dark" key={itemNumber} value={itemNumber} name='delete' onClick={this.handleChange}>
                        <FaTrash/>
                    </button>
                </td> 
            )
        } 
        return
    }

    renderPropertyList(name, items, columns, editButton, deleteButton) {

        return (    
            <div>
                <table className="table table-sm fixed">
                    <thead className="thead-light">
                    
                        <tr>
                            {
                                columns.map((column, i) => (
                                    <th key={i} width={this.getColumnSize(items, editButton, deleteButton)} scope="col">
                                        {column}
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
                                        Object.keys(items[item]).map((row, k) => (
                                            <td key={k}>{items[item][row]}</td>
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

    renderWarning(message) {
        return (
            <div className="alert alert-danger">
                {message}
            </div>
        )
    }

    render() {
        const {name, items, columns, editButton, deleteButton} = this.props
        
        if (items == null) {
            return this.renderWarning('There are no ' + name + ' available')
        } else {           
            return this.renderPropertyList(name, items, columns, editButton, deleteButton)
        }        
    }
}

PropertyList.propTypes = {
    name: PropTypes.string.isRequired,
    items: PropTypes.array.isRequired,
    columns: PropTypes.array.isRequired,
    editButton: PropTypes.bool.isRequired,
    deleteButton: PropTypes.bool.isRequired,
    onEdit: PropTypes.func,
    onDelete: PropTypes.func,
}

export default PropertyList