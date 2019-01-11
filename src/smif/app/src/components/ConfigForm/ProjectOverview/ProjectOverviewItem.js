import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Redirect } from 'react-router-dom'

import {FaTrash, FaPen, FaPlay} from 'react-icons/fa'
import ReactTable from 'react-table'

class ProjectOverviewItem extends Component {
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

    onEditHandler(name) {
        const {itemLink, onClick} = this.props

        if (name != undefined) {
            onClick(itemLink + name)
        }
    }

    onDeleteHandler(name) {
        const {onDelete, itemname} = this.props

        onDelete({
            target: {
                name: itemname,
                value: name,
                type: 'action'
            }
        })
    }

    onStartHandler(name) {
        const {resultLink, onClick} = this.props

        if (name != undefined) {
            onClick(resultLink + name)
        }
    }

    renderItems() {
        const {itemname, items, resultLink, onDelete, onClick} = this.props

        if (this.state.redirect) {
            return <Redirect push to={this.state.redirect_to}/>
        }
        else {
            return (
                <div>
                    <ReactTable
                        data={items}
                        filterable
                        defaultFilterMethod={(filter, row) =>
                            String(row[filter.id]) === filter.value
                        }
                        columns={[{
                            Header: 'Name',
                            accessor: 'name',
                            width: Math.round(window.innerWidth * 0.2),
                            filterMethod: (filter, row) => {
                                return row[filter.id].includes(filter.value)
                            }
                        }, {
                            Header: 'Description',
                            accessor: 'description',
                            width: Math.round(window.innerWidth * 0.3),
                            filterMethod: (filter, row) => {
                                return row[filter.id].includes(filter.value)
                            }
                        }, {
                            Header: 'Actions',
                            width: Math.round(window.innerWidth * 0.1),
                            filterable: false,
                            Cell: row => (
                                <div className='text-center'>
                                    <button
                                        hidden={resultLink==undefined}
                                        id={'btn_start_' + row.value}
                                        type="button"
                                        className="btn btn-outline-dark btn-margin"
                                        name={row.original.name}
                                        onClick={() => this.onStartHandler(row.original.name)}>
                                        <FaPlay/>
                                    </button>
                                    <button
                                        hidden={onClick==undefined}
                                        id={'btn_edit_' + row.original.name}
                                        type="button"
                                        className="btn btn-outline-dark btn-margin"
                                        name={row.original.name}
                                        onClick={() => this.onEditHandler(row.original.name)}>
                                        <FaPen/>
                                    </button>
                                    <button
                                        hidden={onDelete==undefined}
                                        id={'btn_delete_' + row.original.name}
                                        type="button"
                                        className="btn btn-outline-dark btn-margin"
                                        value={itemname}
                                        name={row.original.name}
                                        onClick={() => this.onDeleteHandler(row.original.name)}>
                                        <FaTrash/>
                                    </button>
                                </div>
                            )
                        }]}
                        showPagination={items.length > 50 ? true : false}
                        defaultPageSize={50}
                        showPageSizeOptions={false}
                        minRows={1}
                        className="-striped -highlight"
                        getTdProps={(state, rowInfo, column) => {
                            return {
                                onClick: () => {
                                    if (column.Header != 'Actions') {
                                        this.onEditHandler(rowInfo.original.name)
                                    }
                                },
                                style: {
                                    display: 'flex',
                                    flexDirection: 'column',
                                    justifyContent: 'center'
                                }
                            }
                        }}

                    />
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

    renderInfo(message) {
        return (
            <div id="project_overview_item_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {itemname, items, itemLink } = this.props

        if (itemname == '' || itemname == undefined || itemname == null) {
            return this.renderDanger('There is no itemname configured')
        } else if (itemLink == '' || itemLink == undefined || itemLink == null) {
            return this.renderDanger('There is no itemLink configured')
        } else if (items == null || items == undefined || items.length == 0) {
            return this.renderInfo('There are no items in this list')
        } else {
            return this.renderItems()
        }
    }
}

ProjectOverviewItem.propTypes = {
    itemname: PropTypes.string,
    items: PropTypes.array,
    itemLink: PropTypes.string,
    resultLink: PropTypes.string,
    onDelete: PropTypes.func,
    onClick: PropTypes.func
}

export default ProjectOverviewItem
