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

        this.state = {}
        this.state.createConfig = {name: '', description: ''}
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
        this.props.submit(this.state.createConfig)
    }

    handleCancel() {
        this.props.cancel()
    }

    render() {
        const { createConfig } = this.state
        const { header } = this.props

        console.log(this.state)

        return (
            <div>
                <div className="card">
                    <div className="card-header">{header}</div>
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

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save" onClick={this.handleSubmit} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
            </div>
        )
    }
}

CreateConfigForm.propTypes = {
    header: PropTypes.string,
    submit: PropTypes.func,
    cancel: PropTypes.func
}

export default CreateConfigForm