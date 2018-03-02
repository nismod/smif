import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class DeleteForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
    }

    componentDidMount(){
        document.addEventListener("keydown", this.handleKeyPress, false)
    }

    componentWillUnmount(){
        document.removeEventListener("keydown", this.handleKeyPress, false)
    }

    handleKeyPress(){
        if(event.keyCode === 27) {
            this.handleCancel()
        }
    }

    handleSubmit() {
        this.props.submit(this.props.config_name)
    }

    handleCancel() {
        this.props.cancel()
    }

    render() {
        const { config_name } = this.props

        return (
            <div>
                <div className="card">
                    <div className="card-header">Delete a configuration</div>
                    <div className="card-body">
                        Would you like to delete <b>{config_name}</b>?
                    </div>
                </div>

                <br/>

                <input id="deleteButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Delete" onClick={this.handleSubmit} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
            </div>
        )
    }
}

DeleteForm.propTypes = {
    config_name: PropTypes.string,
    submit: PropTypes.func,
    cancel: PropTypes.func
}

export default DeleteForm