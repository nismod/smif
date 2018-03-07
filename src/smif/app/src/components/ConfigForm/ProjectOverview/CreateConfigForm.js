import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class CreateConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
        this.handleCancel = this.handleCancel.bind(this)
        this.onDismiss = this.onDismiss.bind(this);

        this.state = {}
        this.state.createConfig = {name: '', description: ''}

        this.state.alert_visible = false
        this.state.alert_message = ''
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

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            createConfig: update(this.state.createConfig, {[name]: {$set: value}})
        })
    }

    handleSubmit() {
        const {createConfig} = this.state
        const {existing_names, config_type} = this.props

        if (createConfig.name == '') {
            this.setState({
                alert_message: 'Cannot create a ' + config_type + ' without a name',
                alert_visible: true
            })
        }
        else if (existing_names.includes(createConfig.name)) {
            this.setState({
                alert_message: 'There is already a configuration with the name ' + createConfig.name,
                alert_visible: true
            })
        } else {
            this.props.submit(createConfig)
        }
    }

    handleCancel() {
        this.props.cancel()
    }

    onDismiss() {
        this.setState({alert_visible: false})
    }

    render() {
        const { createConfig } = this.state
        const { config_type } = this.props

        return (
            <div>
                <div className="card">
                    <div className="card-header">Create a new {config_type}</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input autoFocus id="name" className="form-control" name="name" type="text" defaultValue={createConfig.name} onChange={this.handleChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea id="description" className="form-control" name="description" rows="5" defaultValue={createConfig.description} onChange={this.handleChange}/>
                            </div>
                        </div>
                    </div>
                </div>

                <br/>
                
                <div hidden={!this.state.alert_visible} className="alert alert-danger" role="alert">
                    {this.state.alert_message}
                </div>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSubmit} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
            </div>
        )
    }
}

CreateConfigForm.propTypes = {
    config_type: PropTypes.string,
    existing_names: PropTypes.array,
    submit: PropTypes.func,
    cancel: PropTypes.func
}

export default CreateConfigForm