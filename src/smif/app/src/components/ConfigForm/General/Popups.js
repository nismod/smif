import React from 'react'
import PropTypes from 'prop-types'
import Popup from 'components/ConfigForm/General/Popup'
import { DangerButton, CancelButton } from 'components/ConfigForm/General/Buttons'

const ConfirmPopup = (props) => (
    <div>
        <Popup name='bla' onRequestOpen={props.onRequestOpen}>
            <div>
                This form has pending changes. Are you sure you would like to leave without saving?
            </div>
            <br/>
            <div>
                <DangerButton value='Confirm' onClick={props.onConfirm} />
                <CancelButton onClick={props.onCancel}/>
            </div>
        </Popup>
    </div>
)

ConfirmPopup.propTypes = {
    onRequestOpen: PropTypes.bool,
    onConfirm: PropTypes.func,
    onCancel: PropTypes.func
}

export {
    ConfirmPopup
} 