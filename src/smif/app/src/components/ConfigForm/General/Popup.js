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
        transform             : 'translate(-50%, -50%)',
        maxHeight             : '80%',
        overflow              : 'scroll'
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

    componentDidMount() {
        Modal.setAppElement('body')
    }

    render() {
        const {onRequestOpen, name} = this.props

        return (
            <div>
                <Modal isOpen={onRequestOpen} style={customStyles} contentLabel="Example CreateSosModelRunPopup">
                    <div id={name}>
                        <div className="frame">
                            <div className="scroll">
                                {this.props.children}
                            </div>
                        </div>
                    </div>
                </Modal>
            </div>
        )
    }
}

Popup.propTypes = {
    name: PropTypes.string.isRequired,
    onRequestOpen: PropTypes.bool.isRequired,
    children: PropTypes.element.isRequired
}

export default Popup
