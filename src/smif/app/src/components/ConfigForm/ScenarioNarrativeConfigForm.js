import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import { SaveButton, CancelButton } from 'components/ConfigForm/General/Buttons'
import SpecList from './General/SpecList'
import VariantList from './General/VariantList'

class ScenarioConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {
            selected: this.props.scenario_narrative
        }
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.setState({
            selected: update(this.state.selected, {[name]: {$set: value}})
        })
    }

    handleSave() {
        this.props.saveScenario(this.state.selected)
    }

    handleCancel() {
        this.props.cancelScenario()
    }

    render() {
        const {selected} = this.state

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
                                <input id="scenario_name" className="form-control" name="name" type="text" disabled="true" defaultValue={selected.name} onChange={this.handleChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea id="scenario_description" className="form-control" name="description" rows="5" defaultValue={selected.description} onChange={this.handleChange}/>
                            </div>
                        </div>

                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Provides</div>
                    <div className="card-body">        
                        <SpecList 
                            name="provides"
                            specs={selected.provides} 
                            dims={dims} 
                            onChange={this.handleChange} />
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Variants</div>
                    <div className="card-body">        
                        <VariantList 
                            name="variants"
                            variants={selected.variants} 
                            provides={selected.provides} 
                            onChange={this.handleChange} />
                    </div>
                </div>

                <SaveButton onClick={this.handleSave} />
                <CancelButton onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ScenarioConfigForm.propTypes = {
    scenario_narrative: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    saveScenario: PropTypes.func,
    cancelScenario: PropTypes.func
}

export default ScenarioConfigForm
