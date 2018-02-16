import React, { Component } from 'react'
import PropTypes from 'prop-types'

import Modal from 'react-modal'

const customStyles = {
    content : {
        top                   : '50%',
        left                  : '50%',
        right                 : '50%',
        bottom                : 'auto',
        marginRight           : '-50%',
        transform             : 'translate(-50%, -50%)'
    }
}

class Popup extends Component {
    constructor(props) {
        super(props)

        this.state = {
            popupIsOpen: false
        }

        this.openCreateSosModelRunPopup = this.openCreateSosModelRunPopup.bind(this)
        this.closeCreateSosModelRunPopup = this.closeCreateSosModelRunPopup.bind(this)
    }

    openCreateSosModelRunPopup() {
        this.setState({popupIsOpen: true})
    }

    closeCreateSosModelRunPopup() {
        this.setState({popupIsOpen: false})
    }

    componentWillMount() {
        Modal.setAppElement('body')
    }

    render() {
        const {onRequestOpen, onRequestClose} = this.props

        return (
            <div>
                <Modal isOpen={onRequestOpen} style={customStyles} contentLabel="Example CreateSosModelRunPopup">
                    <div>
                        {this.props.children}
                    </div>
                </Modal>
            </div>
        )
    }
}

Popup.propTypes = {
    onRequestOpen: PropTypes.bool.isRequired
}

export default Popup
