import React from 'react'

/**
 * CreateButton
 *
 * Primary button for create actions
 */
const CreateButton = (props) => (
    <input
        id={props.id}
        className="btn btn-success"
        type="button"
        value={props.value? props.value : 'Add'}
        onClick={props.onClick} />
)

/**
 * SaveButton
 *
 * Primary button for save actions
 */
const SaveButton = (props) => (
    <input
        id={props.id}
        className="btn btn-primary"
        type="submit"
        value={props.value? props.value : 'Save'}
        onClick={props.onClick} />
)

/**
 * CancelButton
 *
 * Secondary button for cancel actions
 */
const CancelButton = (props) => (
    <input
        id={props.id}
        className="btn btn-outline-secondary"
        type="button"
        value={props.value? props.value : 'Cancel'}
        onClick={props.onClick} />
)

/**
 * DangerButton
 *
 * Secondary button for cancel actions
 */
const DangerButton = (props) => (
    <input
        id={props.id}
        className="btn btn-danger"
        type="button"
        value={props.value? props.value : 'Delete'}
        onClick={props.onClick} />
)

module.exports = {
    CreateButton,
    SaveButton,
    CancelButton,
    DangerButton
}
