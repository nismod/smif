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
        className="btn btn-success btn-margin"
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
        className="btn btn-primary btn-margin"
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
        className="btn btn-outline-secondary btn-margin"
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
        className="btn btn-danger btn-margin"
        type="button"
        value={props.value? props.value : 'Delete'}
        onClick={props.onClick} />
)

DangerButton.propTypes = {
    id: PropTypes.string,
    value: PropTypes.string,
    onClick: PropTypes.func
}


/** 
 * ToggleButton
 * 
 * Button for toggle actions
 */

const ToggleButton = (props) => (
    <div>
        <button className={'btn ' + (
            (props.active1) ? 'btn-primary active' : 'btn btn-default'
        )}
        onClick={props.action1}>
            {props.label1}</button>
        <button className={'btn ' + (
            (props.active2) ? 'btn-primary active' : 'btn btn-default'
        )}
        onClick={props.action2}>
            {props.label2}</button>
    </div>
)

ToggleButton.propTypes = {
    label1: PropTypes.string,
    label2: PropTypes.string,
    action1: PropTypes.func,
    action2: PropTypes.func,
    active1: PropTypes.bool,
    active2: PropTypes.bool,
}

export {
    CreateButton,
    SaveButton,
    CancelButton,
    DangerButton,
    ToggleButton
}
