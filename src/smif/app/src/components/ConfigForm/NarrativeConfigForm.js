import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import { SaveButton, CancelButton } from 'components/ConfigForm/General/Buttons'
import SpecList from './General/SpecList'
import VariantList from './General/VariantList'

class NarrativeConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selectedNarrative: this.props.narrative
        }
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selectedNarrative: update(this.state.selectedNarrative, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveNarrative(this.state.selectedNarrative)
    }

    handleCancel() {
        this.props.cancelNarrative()
    }

    render() {
        const {selectedNarrative} = this.state

        let dims = this.props.dimensions.map(dim => ({
            value: dim.name,
            label: dim.name
        }))

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input id="narrative_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selectedNarrative.name} onChange={this.handleChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea id="narrative_description" className="form-control" name="description" rows="5" defaultValue={selectedNarrative.description} onChange={this.handleChange}/>
                            </div>
                        </div>

                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Provides</div>
                    <div className="card-body">        
                        <SpecList name="spec" specs={selectedNarrative.provides} dims={dims} />
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Variants</div>
                    <div className="card-body">        
                        <VariantList variants={selectedNarrative.variants} provides={selectedNarrative.provides} />
                    </div>
                </div>

                <SaveButton onClick={this.handleSave} />
                <CancelButton onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

NarrativeConfigForm.propTypes = {
    narrative: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    saveNarrative: PropTypes.func,
    cancelNarrative: PropTypes.func
}

export default NarrativeConfigForm
