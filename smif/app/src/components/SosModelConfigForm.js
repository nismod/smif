import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { Link } from 'react-router-dom'
import update from 'react-addons-update'

import { saveSosModel } from '../actions/actions.js'

import PropertySelector from '../components/PropertySelector.js'
import DependencySelector from '../components/SosModelConfigForm/DependencySelector.js'

class SosModelConfigForm extends Component {
    constructor(props) {
        super(props)

        this.handleSave = this.handleSave.bind(this)
        this.handleCancel = this.handleCancel.bind(this)

        this.state = {}
        this.state.selectedSosModel = this.props.sosModel
    }

    handleSectorModelChange(sectorModel, newState) {
        console.log(sectorModel, newState)
    }

    handleScenarioSetChange(scenarioSet, newState) {
        console.log(scenarioSet, newState)
    }

    handleNarrativeSetChange(narrativeSet, newState) {
        console.log(narrativeSet, newState)
    }

    handleAddDependency(source_model, source_output, sink_model, sink_output) {
        console.log('Add Dependency', source_model, source_output, sink_model, sink_output)
    }

    handleDeleteDependency(id) {
        console.log('Delete Dependency', id)
    }

    handleSave() {
        //console.log(this.state)
        this.props.saveSosModel(this.state.selectedSosModel)
    }

    handleCancel() {
        this.props.cancelSosModel()
    }

    render() {
        const {sosModel, sectorModels, scenarioSets, scenarios, narrativeSets, narratives} = this.props
        const {selectedSosModel} = this.state

        return (
            <div>
                <form>
                    <div className="card">
                        <div className="card-header">General</div>
                        <div className="card-body">

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Name</label>
                                <div className="col-sm-10">
                                    <input className="form-control" name="name" type="text" disabled="true" defaultValue={selectedSosModel.name} onChange={this.handleInputChange}/>
                                </div>
                            </div>

                            <div className="form-group row">
                                <label className="col-sm-2 col-form-label">Description</label>
                                <div className="col-sm-10">
                                    <textarea className="form-control" name="description" rows="5" defaultValue={selectedSosModel.description} onChange={this.handleInputChange}/>
                                </div>
                            </div>

                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Sector Models</div>
                        <div className="card-body">
                            <PropertySelector name="Sector Model" activeProperties={selectedSosModel.sector_models} availableProperties={sectorModels} onChange={this.handleSectorModelChange} />
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Scenario Sets</div>
                        <div className="card-body">
                            <PropertySelector name="Scenario Sets" activeProperties={selectedSosModel.scenario_sets} availableProperties={scenarioSets} onChange={this.handleScenarioSetChange} />
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Narrative Sets</div>
                        <div className="card-body">
                            <PropertySelector name="Narrative Sets" activeProperties={selectedSosModel.narrative_sets} availableProperties={narrativeSets} onChange={this.handleNarrativeSetChange} />
                        </div>
                    </div>

                    <br/>

                    <div className="card">
                        <div className="card-header">Dependencies</div>
                        <div className="card-body">
                            <DependencySelector sosModel={selectedSosModel} sectorModels={sectorModels} onAdd={this.handleAddDependency}/>
                        </div>
                    </div>

                    <br/>

                </form>

                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Save System-of-systems Model Configuration" onClick={this.handleSave} />
                <input className="btn btn-secondary btn-lg btn-block" type="button" value="Cancel" onClick={this.handleCancel} />
            </div>
        )
    }
}

SosModelConfigForm.propTypes = {
    sosModel: PropTypes.object.isRequired,
    sectorModels: PropTypes.array.isRequired,
    scenarioSets: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narrativeSets: PropTypes.array.isRequired,
    saveSosModel: PropTypes.func.isRequired,
    cancelSosModel: PropTypes.func.isRequired
}

export default SosModelConfigForm
            
            
        

            //     <h3>Dependencies</h3>
            //     <div className="table-container">
            //         <table>
            //             <tr>
            //                 <th colSpan="2">Source</th>
            //                 <th colSpan="2">Sink</th>
            //                 <th colSpan="1"></th>
            //             </tr>
            //             <tr>
            //                 <th>Model</th>
            //                 <th>Output</th>
            //                 <th>Model</th>
            //                 <th>Input</th>
            //                 <th></th>
            //             </tr>
            //             <tr>
            //                 <td>population</td>
            //                 <td>count</td>
            //                 <td>energy_demand</td>
            //                 <td>population</td>
            //                 <td><FaTrash /></td>
            //             </tr>
            //             <tr>
            //                 <td>energy_demand</td>
            //                 <td>gas_demand</td>
            //                 <td>energy_supply</td>
            //                 <td>natural_gas_demand</td>
            //                 <td><FaTrash /></td>
            //             </tr>
            //         </table>
            //     </div>

            //     <fieldset>
            //         <label>Source Model:</label>
            //         <div className="select-container">
            //             <select>
            //                 <option value="" disabled="disabled" selected="selected">Select a source model</option>
            //                 <option value="Energy_Demand">Energy Demand</option>
            //                 <option value="Energy_Supply">Energy Supply</option>
            //                 <option value="Transport">Transport</option>
            //                 <option value="Solid_Waste">Solid Waste</option>
            //             </select>
            //         </div>
            //         <label>Source Model Output:</label>
            //         <div className="select-container">
            //             <select>
            //                 <option value="" disabled="disabled" selected="selected">Select a source model output</option>
            //                 <option value="population">Popula            //     <h3>Model</h3>
            //     <fieldset>
            //         <legend>Scenario Sets</legend>
            //         <label>
            //             <input type="checkbox" />
            //             Population
            //         </label>
            //         <label>
            //             <input type="checkbox" />
            //             Economy
            //         </label>
            //     </fieldset>
            //     <fieldset>
            //         <legend>Sector Models</legend>
            //         <label>
            //             <input type="checkbox" />
            //             Energy Dethis.handleSave = this.handleSave.bind(this)mand
            //         </label>
            //         <label>
            //             <input type="checkbox" />
            //             Energy Supply
            //         </label>
            //         <label>
            //             <input type="checkbox" />
            //             Transport
            //         </label>
            //         <label>
            //             <input type="checkbox" />
            //             Solid Waste
            //         </label>
            //     </fieldset>tion</option>
            //                 <option value="total_costs">Total costs</option>
            //                 <option value="fuel_price">Fuel price</option>
            //             </select>
            //         </div>
            //         <label>Sink Model:</label>
            //         <div className="select-container">
            //             <select>
            //                 <option value="" disabled="disabled" selected="selected">Select a sink model</option>
            //                 <option value="Energy_Demand">Energy Demand</option>
            //                 <option value="Energy_Supply">Energy Supply</option>
            //                 <option value="Transport">Transport</option>
            //                 <option value="Solid_Waste">Solid Waste</option>
            //             </select>
            //         </div>
            //         <label>Sink Model Input:</label>
            //         <div className="select-container">
            //             <select>
            //                 <option value="" disabled="disabled" selected="selected">Select a sink model input</option>
            //                 <option value="population">Population</option>
            //                 <option value="total_costs">Total costs</option>
            //                 <option value="fuel_price">Fuel price</option>
            //             </select>
            //         </div>
            //         <input type="button" value="Add Dependency" />
            //     </fieldset>

            //     <input type="button" value="Save SoS Model Configuration" />
            //     <input type="button" value="Cancel" />
            // </div>