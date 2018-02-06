import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'
import NarrativeSetConfigForm from '../../../src/components/ConfigForm/NarrativeSetConfigForm.js'

import {narrative_set} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<NarrativeSetConfigForm />', () => {

    const correctRender = shallow(<NarrativeSetConfigForm narrativeSet={narrative_set} />)
    const dataMissingRender = shallow(<NarrativeSetConfigForm narrativeSet={empty_object} />)

    it('renders narrative_set.name', () => {
        const narrative_set_name = correctRender.find('[id="narrative_set_name"]')
        expect(narrative_set_name.html()).to.contain(narrative_set.name)
    })

    it('renders narrative_set.name when data missing', () => {
        const narrative_set_name = dataMissingRender.find('[id="narrative_set_name"]')
        expect(narrative_set_name.html()).to.contain(`id="narrative_set_name"`)
    })

    it('renders narrative_set.description', () => {
        const narrative_set_description = correctRender.find('[id="narrative_set_description"]')
        expect(narrative_set_description.html()).to.contain(narrative_set.description)
    })

    it('renders narrative_set.description when data missing', () => {
        const narrative_set_description = dataMissingRender.find('[id="narrative_set_description"]')
        expect(narrative_set_description.html()).to.contain(`id="narrative_set_description"`)
    })

    it('loads properties ', () => {
        const wrapper = mount((<NarrativeSetConfigForm narrativeSet={narrative_set} />))
        expect(wrapper.props()['narrativeSet']).to.equal(narrative_set)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<NarrativeSetConfigForm narrativeSet={narrative_set} saveNarrativeSet={onSaveClick} />))
    
        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.equal(narrative_set)
    })

    it('save callback with changed properties', () => {
        const changed_narrative_set = {
            name: 'new_name',
            description: 'new_description'
        }

        const onSaveClick = sinon.spy()
        const wrapper = mount((<NarrativeSetConfigForm narrativeSet={narrative_set} saveNarrativeSet={onSaveClick} />))

        wrapper.find('[id="narrative_set_name"]').simulate('change', { target: { name: 'name', value: changed_narrative_set['name'] } })
        wrapper.find('[id="narrative_set_description"]').simulate('change', { target: { name: 'description', value: changed_narrative_set['description'] } })
        wrapper.find('[id="saveButton"]').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_narrative_set)
    })
    
    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<NarrativeSetConfigForm narrativeSet={narrative_set} cancelNarrativeSet={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<NarrativeSetConfigForm narrativeSet={narrative_set} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
