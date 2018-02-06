import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'

import NarrativeConfigForm from '../../../src/components/ConfigForm/NarrativeConfigForm.js'

import {narrative, narrative_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<NarrativeConfigForm />', () => {

    const correctRender = shallow(<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} />)
    const dataMissingRender = shallow(<NarrativeConfigForm narrative={empty_object} narrativeSets={empty_array} />)
    
    it('renders narrative.name', () => {
        const narrative_name = correctRender.find('[id="narrative_name"]')
        expect(narrative_name.html()).to.contain(narrative.name)
    })

    it('renders narrative.name when data missing', () => {
        const narrative_name = dataMissingRender.find('[id="narrative_name"]')
        expect(narrative_name.html()).to.contain(`id="narrative_name"`)
    })
    
    it('renders narrative.description', () => {
        const narrative_description = correctRender.find('[id="narrative_description"]')
        expect(narrative_description.html()).to.contain(narrative.description)
    })
    
    it('renders narrative.description when data missing', () => {
        const narrative_description = dataMissingRender.find('[id="narrative_description"]')
        expect(narrative_description.html()).to.contain(`id="narrative_description"`)
    })

    it('loads properties ', () => {
        const wrapper = mount((<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} />))
        expect(wrapper.props()['narrative']).to.equal(narrative)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} saveNarrative={onSaveClick} />))
    
        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.equal(narrative)
    })

    it('save callback with changed properties', () => {
        const changed_narrative = {
            name: 'new_narrative_name',
            description: 'new_description',
            filename: 'new_filename',
            narrative_set: 'new_narrative_set'
        }

        const onSaveClick = sinon.spy()
        const wrapper = mount((<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} saveNarrative={onSaveClick} />))

        wrapper.find('[id="narrative_name"]').simulate('change', { target: { name: 'name', value: changed_narrative['name'] } })
        wrapper.find('[id="narrative_name"]').simulate('change', { target: { name: 'description', value: changed_narrative['description'] } })
        wrapper.find('[id="narrative_name"]').simulate('change', { target: { name: 'filename', value: changed_narrative['filename'] } })
        wrapper.find('[id="narrative_name"]').simulate('change', { target: { name: 'narrative_set', value: changed_narrative['narrative_set'] } })
        wrapper.find('[id="saveButton"]').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_narrative)
    })
    
    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} cancelNarrative={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
