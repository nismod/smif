import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import { PrimaryButton, SecondaryButton } from 'components/ConfigForm/General/Buttons'
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

    handleChange(key, value) {
        this.props.onEdit()

        this.setState({
            selected: update(this.state.selected, {[key]: {$set: value}})
        })
    }

    handleSave() {
        this.props.onSave(this.state.selected)
        this.props.onCancel()
    }

    handleCancel() {
        this.props.onCancel()
    }

    render() {
        const {selected} = this.state

        if (this.props.save) {
            this.props.onSave(this.state.selected)
        }

        let dims = this.props.dimensions.map(dim => ({
            value: dim.name,
            label: dim.name
        }))

        dims.sort(function(a, b){
            if(a.label < b.label) { return -1 }
            if(a.label > b.label) { return 1 }
            return 0
        })

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input 
                                    className="form-control" 
                                    type="text" 
                                    disabled="true" 
                                    defaultValue={selected.name} 
                                    onChange={(event) => this.handleChange('name', event.target.value)}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea 
                                    className="form-control" 
                                    rows="5" 
                                    defaultValue={selected.description} 
                                    onChange={(event) => this.handleChange('description', event.target.value)}/>
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
                            onChange={(event) => this.handleChange('provides', event.target.value)} />
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Variants</div>
                    <div className="card-body">        
                        <VariantList 
                            variants={selected.variants} 
                            provides={selected.provides} 
                            require_provide_full_config={this.props.require_provide_full_variant}
                            onChange={(event) => this.handleChange('description', event.target.value)} />
                    </div>
                </div>

                <PrimaryButton value="Save" onClick={this.handleSave} />
                <SecondaryButton value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ScenarioConfigForm.propTypes = {
    scenario_narrative: PropTypes.object.isRequired,
    dimensions: PropTypes.array.isRequired,
    saveScenarioNarrative: PropTypes.func,
    require_provide_full_variant: PropTypes.bool,
    cancelScenarioNarrative: PropTypes.func,
    save: PropTypes.bool,
    onSave: PropTypes.func,
    onCancel: PropTypes.func,
    onEdit: PropTypes.func
}

ScenarioConfigForm.defaultValue = {
    require_provide_full_variant: false
}

export default ScenarioConfigForm
