import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSosModels } from '../actions/actions.js'
import { fetchScenarios } from '../actions/actions.js'
import { fetchNarratives } from '../actions/actions.js'

import { resetSosModelRun } from '../actions/actions.js'
import { saveSosModelRun } from '../actions/actions.js'

import SosModelConfigForm from '../components/SosModelConfigForm.js'

import FaEdit from 'react-icons/lib/fa/edit';
import FaTrash from 'react-icons/lib/fa/trash';

class SosModelConfig extends Component {
    componentDidMount() {
        const {dispatch } = this.props

        dispatch(fetchSosModel(this.props.match.params.name))
        dispatch(fetchScenarios())
        dispatch(fetchNarratives())

        this.saveSosModel = this.saveSosModel.bind(this)
        this.returnToPreviousPage = this.returnToPreviousPage.bind(this)
    }

    componentWillUnmount() {
        const { dispatch } = this.props
        dispatch(resetSosModel())
    }

    saveSosModelRun(sosModel) {
        const { dispatch } = this.props
        dispatch(saveSosModel(sosModel))
        this.returnToPreviousPage()
    }

    returnToPreviousPage() {
        history.back()
    }

    render() {
        const {sos_model_run, sos_models, scenarios, narratives, isFetching} = this.props


        return (
    
            <div>
                <h1>Model Configuration</h1>

                <div hidden={ !isFetching } className="alert alert-primary">
                    Loading...
                </div>

                <div hidden className="alert alert-danger">
                    Error
                </div>

                <div hidden={ isFetching }>           
{/* 
                    {config_form}            
                    {buttons} */}

                </div>
            </div>

            // <div className="content-wrapper">
            //     <h2>System-of-Systems Model Configuration</h2>

            //     <h3>General</h3>
            //     <label>Name:</label>
            //     <input type="text" name="model_name"  value="sos_model_name"/>
            //     <label>Description:</label>
            //     <div className="textarea-container">
            //         <textarea name="textarea" rows="5" value="A system of systems model which encapsulates the future supply and demand of energy for the UK"/>
            //     </div>

            //     <h3>Model</h3>
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
            //             Energy Demand
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
            //     </fieldset>

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
            //                 <option value="population">Population</option>
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
        )
    }
}

SosModelConfig.propTypes = {
    sos_model_run: PropTypes.object.isRequired,
    sos_models: PropTypes.array.isRequired,
    scenarios: PropTypes.array.isRequired,
    narratives: PropTypes.array.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
}

function mapStateToProps(state) {
    return {
        sos_model_run: state.sos_model_run.item,
        sos_models: state.sos_models.items,
        scenarios: state.scenarios.items,
        narratives: state.narratives.items,
        isFetching: state.sos_model_run.isFetching
    }
}

export default connect(mapStateToProps)(SosModelConfig)
