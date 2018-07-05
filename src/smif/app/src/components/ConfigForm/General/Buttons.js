import React from 'react'
import PropTypes from 'prop-types'

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

CreateButton.propTypes = {
    id: PropTypes.string,
    value: PropTypes.string,
    onClick: PropTypes.func
}

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

SaveButton.propTypes = {
    id: PropTypes.string,
    value: PropTypes.string,
    onClick: PropTypes.func
}

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

CancelButton.propTypes = {
    id: PropTypes.string,
    value: PropTypes.string,
    onClick: PropTypes.func
}

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

DangerButton.propTypes = {
    id: PropTypes.string,
    value: PropTypes.string,
    onClick: PropTypes.func
}

export {
    CreateButton,
    SaveButton,
    CancelButton,
    DangerButton
}
