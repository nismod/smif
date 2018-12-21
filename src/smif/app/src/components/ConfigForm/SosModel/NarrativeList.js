import React, { Component } from 'react'
import PropTypes from 'prop-types'
import update from 'immutability-helper'

import Popup from 'components/ConfigForm/General/Popup.js'
import PropertySelector from 'components/ConfigForm/General/PropertySelector.js'
import Select from 'react-select'
import {PrimaryButton, SecondaryButton, SuccessButton, DangerButton} from 'components/ConfigForm/General/Buttons'

class NarrativeList extends Component {
    constructor(props) {
        super(props)

        this.state = {
            formNarrativePopupIsOpen: false,
            formVariantPopupIsOpen: false,
            formEditMode: false,
            narrative: this.emptyNarrative(),
            variant: this.emptyVariant()
        }

        this.handleNarrativeFormInput = this.handleNarrativeFormInput.bind(this)
        this.handleVariantFormInput = this.handleVariantFormInput.bind(this)
        this.handleCreateNarrative = this.handleCreateNarrative.bind(this)
        this.handleCreateVariant = this.handleCreateVariant.bind(this)
        this.handleNarrativeEdit = this.handleNarrativeEdit.bind(this)
        this.handleVariantEdit = this.handleVariantEdit.bind(this)
        this.handleNarrativeSubmit = this.handleNarrativeSubmit.bind(this)
        this.handleVariantSubmit = this.handleVariantSubmit.bind(this)
        this.handleNarrativeDelete = this.handleNarrativeDelete.bind(this)
        this.handleVariantDelete = this.handleVariantDelete.bind(this)
    }

    openNarrativeForm() {
        this.setState({formNarrativePopupIsOpen: true})
    }

    openVariantForm() {
        this.setState({formVariantPopupIsOpen: true})
    }

    closeNarrativeForm() {
        this.setState({formNarrativePopupIsOpen: false})
        this.setState({formEditMode: false})
    }

    closeVariantForm() {
        this.setState({formVariantPopupIsOpen: false})
        this.setState({formEditMode: false})
    }

    handleNarrativeFormInput(data, var1, var2=undefined) {

        if (var1 == 'name' || var1 == 'description') {
            this.setState({
                narrative: update(this.state.narrative, {[var1]: {$set: data}})
            })
        } else {
            let narrative = JSON.parse(JSON.stringify(this.state.narrative))
            narrative.provides[var2] = data

            if (narrative.provides[var2].length == 0){
                delete narrative.provides[var2]
            }

            this.setState({
                narrative: narrative
            })
        }
    }

    handleVariantFormInput(data, var1, var2=undefined) {

        if (var1 == 'name' || var1 == 'description') {
            this.setState({
                variant: update(this.state.variant, {[var1]: {$set: data}})
            })
        } else {
            let variant = JSON.parse(JSON.stringify(this.state.variant))
            variant.data[var2] = data

            this.setState({
                variant: variant
            })
        }
    }

    handleNarrativeSubmit(event) {
        event.preventDefault()

        // Sync provide with variants
        let parameters = []
        Object.keys(this.state.narrative.provides).forEach(sos_model => {
            this.state.narrative.provides[sos_model].forEach(parameter => {
                parameters.push(parameter)
            })
        })

        this.state.narrative.variants.forEach(variant => {

            // Remove parameter definition that are not provided
            Object.keys(variant.data).forEach(parameter => {
                if (!parameters.includes(parameter)) {
                    delete variant.data[parameter]
                }
            })

            // Add parameters that are provided but not defined
            parameters.forEach(parameter => {
                if (!Object.keys(variant.data).includes(parameter)) {
                    variant.data[parameter] = ''
                }
            })
        })

        // update narratives
        let index = this.props.narratives.findIndex(narrative => narrative.name === this.state.narrative.name)
        let new_narratives =JSON.parse(JSON.stringify(this.props.narratives))
        if (index >= 0) {
            new_narratives.splice(index, 1, this.state.narrative)
        }
        else {
            new_narratives.push(this.state.narrative)
        }
        this.props.onChange(new_narratives)
        this.closeNarrativeForm()
    }

    handleVariantSubmit(event) {
        event.preventDefault()

        let narrative = JSON.parse(JSON.stringify(this.state.narrative))

        // update variant
        let index = narrative.variants.findIndex(variant => variant.name === this.state.variant.name)
        if (index >= 0) {
            narrative.variants.splice(index, 1, this.state.variant)
        }
        else {
            narrative.variants.push(this.state.variant)
        }

        // update narrative
        let new_narratives = JSON.parse(JSON.stringify(this.props.narratives))
        index = this.props.narratives.findIndex(prop_narrative => prop_narrative.name === narrative.name)
        new_narratives.splice(index, 1, narrative)
        this.props.onChange(new_narratives)
        this.closeVariantForm()
    }

    emptyNarrative() {
        return {
            name: '',
            description: '',
            sos_model: '',
            provides: {},
            variants: []
        }
    }

    emptyVariant(narrative_name=undefined) {
        if (narrative_name==undefined) {
            return {
                name: '',
                description: '',
                data: {}
            }
        } else {
            let parameters = []
            let narrative = this.props.narratives.filter(narrative => narrative.name == narrative_name)[0]
            Object.keys(narrative.provides).forEach(sos_model => {
                narrative.provides[sos_model].forEach(parameter => {
                    parameters.push(parameter)
                })
            })

            return {
                name: '',
                description: '',
                data: parameters.reduce((obj, item) => {
                    obj[item] = ''
                    return obj
                }, {})
            }
        }
    }

    handleCreateNarrative() {
        this.setState({narrative: this.emptyNarrative()})
        this.openNarrativeForm()
    }

    handleCreateVariant(narrative_name=undefined) {

        if (narrative_name != undefined) {
            let selectedNarrative = JSON.parse(JSON.stringify(this.props.narratives.filter(narrative => narrative.name == narrative_name)[0]))

            this.setState({narrative: {
                ...this.emptyNarrative(),
                ...selectedNarrative}
            })
        }

        this.setState({variant: this.emptyVariant(narrative_name)})
        this.openVariantForm()
    }

    handleNarrativeEdit(event) {
        const target = event.currentTarget
        const name = target.dataset.name

        let selectedNarrative = JSON.parse(JSON.stringify(this.props.narratives.filter(narrative => narrative.name == name)[0]))

        this.setState({narrative: {
            ...this.emptyNarrative(),
            ...selectedNarrative}
        })
        this.setState({formEditMode: true})
        this.openNarrativeForm()
    }

    handleVariantEdit(event) {

        const target = event.currentTarget
        const narrative_name = target.dataset.name.split(',')[0]
        const variant_name = target.dataset.name.split(',')[1]

        let selectedNarrative =  JSON.parse(JSON.stringify(this.props.narratives.filter(narrative => narrative.name == narrative_name)[0]))
        let selectedVariant =  JSON.parse(JSON.stringify(selectedNarrative.variants.filter(variant => variant.name == variant_name)[0]))

        this.setState({narrative: {
            ...this.emptyNarrative(),
            ...selectedNarrative}
        })

        this.setState({variant: {
            ...this.emptyVariant(),
            ...selectedVariant}
        })

        this.setState({formEditMode: true})
        this.openVariantForm()
    }

    handleNarrativeDelete(name) {
        let new_narratives = JSON.parse(JSON.stringify(this.props.narratives))
        new_narratives.splice(this.props.narratives.findIndex(narrative => narrative.name === name), 1)
        this.props.onChange(new_narratives)
        this.closeNarrativeForm()
    }

    handleVariantDelete() {

        let narrative = JSON.parse(JSON.stringify(this.state.narrative))

        // update variant
        let index = narrative.variants.findIndex(variant => variant.name === this.state.variant.name)
        narrative.variants.splice(index, 1)

        // update narrative
        let new_narratives = JSON.parse(JSON.stringify(this.props.narratives))
        index = this.props.narratives.findIndex(prop_narrative => prop_narrative.name === narrative.name)
        new_narratives.splice(index, 1, narrative)
        this.props.onChange(new_narratives)
        this.closeVariantForm()
    }

    renderNarrativeList(name, narratives) {
        var columns = []
        columns.push('Narrative')
        columns.push('Variants')

        return (
            <div>
                <table className="table table-hover table-list">
                    <thead className="thead-light">
                        <tr>
                            {
                                columns.map((column) => (
                                    <th className="col-text"
                                        scope="col" key={name + '_column_' + column}>
                                        {column}
                                    </th>
                                ))
                            }
                        </tr>
                    </thead>
                    <tbody>
                        {
                            narratives.map((narrative) => (
                                <tr key={narrative.name}>
                                    <td className="col-text"
                                        data-name={narrative.name}
                                        onClick={(e) => this.handleNarrativeEdit(e)}>
                                        <div title={narrative.description}>
                                            {narrative.name}
                                        </div>
                                    </td>
                                    <td className="col-text">
                                        {
                                            narrative.variants.map((variant, idx) => (
                                                <button key={idx}
                                                    type="button"
                                                    data-name={[narrative.name, variant.name]}
                                                    className="btn btn-margin btn-outline-secondary"
                                                    onClick={(e) => this.handleVariantEdit(e)}>
                                                    {variant.name}
                                                </button>
                                            ))
                                        }
                                        <button
                                            type="button"
                                            className="btn btn-margin btn-outline-secondary"
                                            onClick={(e) => this.handleCreateVariant(narrative.name)}>
                                            {'+'}
                                        </button>
                                    </td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>

                <SuccessButton id={'btn_add_' + name} value={'Add ' + name} onClick={() => this.handleCreateNarrative()} />
                <Popup name={'popup_add_' + name} onRequestOpen={this.state.formNarrativePopupIsOpen}>
                    <form className="form-config" onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleNarrativeSubmit(e)}}>
                        <div>
                            <div className="container">
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Name</label>
                                        <input
                                            id={name + '_narrative_name'}
                                            className='form-control'
                                            type="text"
                                            name='name'
                                            disabled={this.state.formEditMode}
                                            value={this.state.narrative.name}
                                            onChange={(e) => this.handleNarrativeFormInput(e.target.value, 'name')}
                                            autoFocus
                                            required
                                        />
                                    </div>
                                    <div className="col">
                                        <label>Description</label>
                                        <input
                                            id={name + '_narrative_description'}
                                            className='form-control'
                                            type="text"
                                            name="description"
                                            value={this.state.narrative.description}
                                            onChange={(e) => this.handleNarrativeFormInput(e.target.value, 'description')}
                                            required
                                        />
                                    </div>
                                </div>
                                <div className="row">
                                    <div className="col">
                                        <label className='label'>Provides</label>
                                        {
                                            Object.keys(this.state.narrative.provides).concat(
                                                this.props.sector_models.map(
                                                    sector_model => sector_model.name)).filter(
                                                    function(item, i, ar){ return ar.indexOf(item) === i; }
                                                ).sort().map((sector_model, idx) => (
                                                <div key={idx}>
                                                    <span className="badge badge badge-secondary">{sector_model}</span>
                                                    <PropertySelector
                                                        name={sector_model}
                                                        activeProperties={
                                                            sector_model in this.state.narrative.provides
                                                                ? this.state.narrative.provides[sector_model]
                                                                : []
                                                            }
                                                        availableProperties={
                                                            this.props.sector_models.filter(sectormodel => sectormodel.name == sector_model).length > 0
                                                                ? this.props.sector_models.filter(sectormodel => sectormodel.name == sector_model)[0].parameters
                                                                : []
                                                            }
                                                        onChange={(e) => this.handleNarrativeFormInput(e.target.value, 'provide', sector_model)} />
                                                    <br/>
                                                </div>
                                            ))
                                        }
                                    </div>
                                </div>
                            </div>

                            <PrimaryButton id={'btn_' + name + '_save'} value="Save" />
                            <SecondaryButton id={'btn_' + name + '_cancel'} value="Cancel" onClick={() => this.closeNarrativeForm()}/>
                            {
                                !this.state.formEditMode ? null : (
                                    <DangerButton
                                        id={'btn_' + name + '_delete'}
                                        onClick={() => this.handleNarrativeDelete(this.state.narrative.name)} />
                                )
                            }
                        </div>
                    </form>
                </Popup>

                <Popup name={'popup_variant_' + name} onRequestOpen={this.state.formVariantPopupIsOpen}>
                    <form className="form-config" onSubmit={(e) => {e.preventDefault(); e.stopPropagation(); this.handleVariantSubmit(e)}}>
                        <div>
                            <div className="container">
                                <div className="row">
                                    <div className="col-4">
                                        <label className='label'>Name</label>
                                        <input
                                            id={name + '_variant_name'}
                                            className='form-control'
                                            type="text"
                                            name='name'
                                            disabled={this.state.formEditMode}
                                            value={this.state.variant.name}
                                            onChange={(e) => this.handleVariantFormInput(e.target.value, 'name')}
                                            autoFocus
                                            required
                                        />
                                    </div>
                                    <div className="col-8">
                                        <label>Description</label>
                                        <input
                                            id={name + '_variant_description'}
                                            className='form-control'
                                            type="text"
                                            name="description"
                                            value={this.state.variant.description}
                                            onChange={(e) => this.handleVariantFormInput(e.target.value, 'description')}
                                            required
                                        />
                                    </div>
                                </div>

                                <div className="row">
                                    <div className="col-4">
                                        <label className='label'>Parameter</label>
                                        {
                                            Object.keys(this.state.variant.data).map((variant, idx) => (
                                                <div key={idx}>
                                                    <input
                                                        id={name + '_variant_name'}
                                                        className='form-control'
                                                        type="text"
                                                        name='parameter'
                                                        disabled={true}
                                                        value={variant}
                                                        onChange={this.handleVariantFormInput}
                                                        autoFocus
                                                        required
                                                    />
                                                </div>
                                            ))
                                        }
                                    </div>
                                    <div className="col-8">
                                        <label className='label'>Datafile</label>
                                        {
                                            Object.keys(this.state.variant.data).map((parameter, idx) => (
                                                <div key={idx}>
                                                    <input
                                                        id={name + '_variant_description'}
                                                        className='form-control'
                                                        type="text"
                                                        name="description"
                                                        value={this.state.variant.data[parameter]}
                                                        onChange={(e) => this.handleVariantFormInput(e.target.value, 'parameter', parameter)}
                                                        required
                                                    />
                                                </div>
                                            ))
                                        }
                                    </div>
                                </div>
                            </div>

                            <br/>

                            <PrimaryButton id={'btn_' + name + '_save'} value="Save" />
                            <SecondaryButton id={'btn_' + name + '_cancel'} value="Cancel" onClick={() => this.closeVariantForm()}/>
                            {
                                !this.state.formEditMode ? null : (
                                    <DangerButton
                                        id={'btn_' + name + '_delete'}
                                        onClick={() => this.handleVariantDelete()} />
                                )
                            }
                        </div>
                    </form>
                </Popup>
            </div>
        )
    }

    render() {
        const {name, narratives, sector_models} = this.props

        return this.renderNarrativeList(name, narratives, sector_models)
    }
}

NarrativeList.propTypes = {
    name: PropTypes.string.isRequired,
    narratives: PropTypes.array.isRequired,
    sector_models: PropTypes.array.isRequired,
    onChange: PropTypes.func
}

export default NarrativeList
