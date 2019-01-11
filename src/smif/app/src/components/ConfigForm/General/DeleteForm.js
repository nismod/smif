import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { FaPen } from 'react-icons/fa'
import { SecondaryButton, DangerButton } from './Buttons'

class DeleteForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
        this.handleCancel = this.handleCancel.bind(this)
    }

    componentDidMount(){
        document.addEventListener('keydown', this.handleKeyPress, false)
    }

    componentWillUnmount(){
        document.removeEventListener('keydown', this.handleKeyPress, false)
    }

    handleKeyPress(event){
        if(event.keyCode === 27) {
            this.handleCancel()
        }
        if(event.keyCode === 13) {
            this.handleSubmit()
        }
    }

    handleSubmit() {
        this.props.submit(this.props.config_name)
    }

    handleCancel() {
        this.props.cancel()
    }

    render() {
        const { config_name, config_type, in_use_by } = this.props

        if (in_use_by == undefined || in_use_by.length == 0) {
            return (
                <div>
                    <div className="card">
                        <div className="card-header">Delete a configuration</div>
                        <div className="card-body">
                            Would you like to delete the <b>{config_type}</b> with
                            name <b>{config_name}</b>?
                        </div>
                    </div>

                    <DangerButton id="deleteButton" value="Delete" onClick={this.handleSubmit} />
                    <SecondaryButton id="cancelDelete" value="Cancel" onClick={this.handleCancel} />
                </div>
            )
        } else {
            return (
                <div>
                    <div className="card">
                        <div className="card-header">Delete a configuration</div>
                        <div className="card-body">
                            <p>It is not possible to delete <b>{config_type}</b> with name <b>{config_name}</b></p>
                            <p>Because it is in use by the following configurations:</p>
                            <ul>
                                {in_use_by.map((conf) =>
                                    <div key={conf.name}>
                                        <button
                                            id={'btn_edit_' + conf.name}
                                            type="button"
                                            className="btn btn-outline-dark btn-margin"
                                            name={conf.name}
                                            onClick={() => this.props.onClick(conf.link + conf.name)}>
                                            <FaPen/>
                                        </button>
                                        {conf.name}
                                        ({conf.type})
                                    </div>

                                )}
                            </ul>
                        </div>
                    </div>

                    <SecondaryButton id="cancelDelete" value="Cancel" onClick={this.handleCancel} />
                </div>
            )
        }
    }
}

DeleteForm.propTypes = {
    config_name: PropTypes.string,
    config_type: PropTypes.string,
    in_use_by: PropTypes.array,
    onClick: PropTypes.func,
    submit: PropTypes.func,
    cancel: PropTypes.func
}

export default DeleteForm
