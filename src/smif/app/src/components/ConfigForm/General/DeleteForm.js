import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import FaPencil from 'react-icons/lib/fa/pencil'

import { Redirect } from 'react-router-dom'

class DeleteForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleRedirect = this.handleRedirect.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            redirect: false,
            redirect_to: ''
        }
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

    handleRedirect(link, configuration) {
        this.setState({
            redirect: true,
            redirect_to: link + configuration
        })
    }

    handleSubmit() {
        this.props.submit(this.props.config_name)
    }

    handleCancel() {
        this.props.cancel()
    }

    render() {
        const { config_name, config_type, in_use_by } = this.props

        let message = (<div></div>)

        if (isNaN(config_name)) {
            message = (<div>Would you like to delete the <b>{config_type}</b> with name <b>{config_name}</b>?</div>)
        } else {
            message = (<div>Would you like to delete this <b>{config_type}</b>?</div>)
        }

        if (this.state.redirect) {
            return <Redirect push to={this.state.redirect_to}/>
        } else if (in_use_by == undefined || in_use_by.length == 0) {
            return (
                <div>
                    <div className="card">
                        <div className="card-header">Delete a configuration</div>
                        <div className="card-body">
                            {message}
                        </div>
                    </div>

                    <br/>

                    <input autoFocus id="deleteButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Delete" onClick={this.handleSubmit} />
                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
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
                                {in_use_by.map((configuration) =>
                                    <div key={configuration.name}>
                                        <button onClick={() => this.handleRedirect(configuration.link, configuration.name)} type="button" className="btn btn-outline-dark">
                                            <FaPencil/>
                                        </button>
                                        <a> {configuration.name + ' (' + configuration.type + ')'}</a>
                                    </div>
                                )}
                            </ul>
                        </div>
                    </div>

                    <br/>

                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
                </div>
            )
        }
    }
}

DeleteForm.propTypes = {
    config_name: PropTypes.string,
    config_type: PropTypes.string,
    in_use_by: PropTypes.array,
    submit: PropTypes.func,
    cancel: PropTypes.func
}

export default DeleteForm