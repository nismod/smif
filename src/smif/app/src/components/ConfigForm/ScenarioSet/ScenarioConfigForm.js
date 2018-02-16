import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import PropertyList from '../General/PropertyList'

class ScenarioConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleKeyPress = this.handleKeyPress.bind(this)
        this.handleFacetChange = this.handleFacetChange.bind(this)
        this.handleChange = this.handleChange.bind(this)
        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedScenario = this.props.scenario
        this.state.currentFacet = 0
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

    handleFacetChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        console.log(name)
        console.log(value)

        this.state.currentFacet = name
        this.forceUpdate()
    }

    handleChange(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        if (name.startsWith('facet_')) {
            this.state.selectedScenario.facets[this.state.currentFacet][name.slice(6)] = value
        } else {
            this.state.selectedScenario[name] = value
        }

        this.forceUpdate()
    }

    handleSave() {
        this.props.saveScenario(this.state.selectedScenario)
    }

    handleCancel() {
        this.props.cancelScenario()
    }

    render() {
        const {selectedScenario, currentFacet} = this.state
        const {scenario} = this.props

        console.log(selectedScenario)

        let editMode = true
        if (scenario.name === undefined) editMode = false

        let facetNav = []
        for (let i=0; i < selectedScenario.facets.length; i++) {
            facetNav.push(<li className={'page-item-' + i} key={i}><a className="page-link" name={i} onClick={this.handleFacetChange}>{i+1}</a></li>)
        }

        return (
            <div>
                <div className="card">
                    <div className="card-header">General</div>
                    <div className="card-body">

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Name</label>
                            <div className="col-sm-10">
                                <input id="scenario_name" className="form-control" name="name" type="text" disabled={editMode} defaultValue={selectedScenario.name} onChange={this.handleChange}/>
                            </div>
                        </div>

                        <div className="form-group row">
                            <label className="col-sm-2 col-form-label">Description</label>
                            <div className="col-sm-10">
                                <textarea id="scenario_description" className="form-control" name="description" rows="5" defaultValue={selectedScenario.description} onChange={this.handleChange}/>
                            </div>
                        </div>

                    </div>
                </div>

                <br/>
                    
                <div className="card">
                    <div className="card-header">Facets</div>
                    <div className="card-body">
                        <div className="container">
                            <div className="row">
                                <div className="col">
                                    <label>Name</label>
                                    <input autoFocus className='form-control' type="text" name="facet_name" value={selectedScenario.facets[currentFacet].name} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="row">
                                <div className="col">
                                    <label>Filename</label>
                                </div>
                                <div className="col">
                                    <label>Units</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input type="text" className='form-control' name="facet_filename" value={selectedScenario.facets[currentFacet].filename} onChange={this.handleChange}/>
                                </div>
                                <div className="col">
                                    <input type="text" className='form-control' name="facet_units" value={selectedScenario.facets[currentFacet].units} onChange={this.handleChange}/>
                                </div>
                            </div>

                            <div className="row">
                                <div className="col">
                                    <label>Spatial Resolution</label>
                                </div>
                                <div className="col">
                                    <label>Temporal Resolution</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="col">
                                    <input type="text" className='form-control' name="facet_spatial_resolution" value={selectedScenario.facets[currentFacet].spatial_resolution} onChange={this.handleChange}/>
                                </div>
                                <div className="col">
                                    <input type="text" className='form-control' name="facet_temporal_resolution" value={selectedScenario.facets[currentFacet].temporal_resolution} onChange={this.handleChange}/>
                                </div>
                            </div>
                        </div>
                        <br/>

                        <nav>
                            <ul className="pagination justify-content-center"> 
                                {facetNav}
                            </ul>
                        </nav>
                    </div>
                </div>

                <br/>

                <input id="saveButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Save Scenario" onClick={this.handleSave} />
                <input id="cancelButton" className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />

                <br/>
            </div>
        )
    }
}

ScenarioConfigForm.propTypes = {
    scenario: PropTypes.object.isRequired,
    scenarioSet: PropTypes.object.isRequired,
    saveScenario: PropTypes.func,
    cancelScenario: PropTypes.func
}

export default ScenarioConfigForm
