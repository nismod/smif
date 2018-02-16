import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

class FacetConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedFacet = this.props.facet
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
            selectedFacet: update(this.state.selectedFacet, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveFacet(this.state.selectedFacet)
    }

    handleCancel() {
        this.props.cancelFacet()
    }

    render() {
        const { selectedFacet } = this.state
        const { facet } = this.props

        let editMode = true
        if (Object.keys(facet).length === 0) editMode = false

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input id="facet_name" className="form-control" name="name" type="text" disabled={editMode} defaultValue={selectedFacet.name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea id="facet_description" className="form-control" name="description" rows="5" defaultValue={selectedFacet.description} onChange={this.handleChange}/>
                                </div>
                            </div>

                        </div>

                    </div>

                    <br/>

                    <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save Facet" onClick={this.handleSave} />
                    <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
                </form>
            </div>
        )
    }
}

FacetConfigForm.propTypes = {
    facet: PropTypes.object,
    saveFacet: PropTypes.func,
    cancelFacet: PropTypes.func
}

export default FacetConfigForm

// <div className="form-group row">
// <label className="col-sm-2 col-form-label">Name</label>
// <div className="col-sm-10">
//     <input autoFocus className="form-control" name="createPopupName" type="text" onChange={this.handleInputChange} required/>
// </div>
// </div>

// <input className="btn btn-secondary btn-lg btn-block" type="submit" value="Create"/>
// <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.closeCreatePopup}/>