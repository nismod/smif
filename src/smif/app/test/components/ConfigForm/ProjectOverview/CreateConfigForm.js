import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'

import CreateConfigForm from '../../../../src/components/ConfigForm/ProjectOverview/CreateConfigForm.js'

describe('<CreateConfigForm />', () => {

    it('renders config_type', () => {
        const form = shallow(<CreateConfigForm config_type={'testconfig'} existing_names={[]} />)
        expect(form.html()).to.contain('<div class="card-header">Create a new testconfig</div>')
    })

    it('create new configuration', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<CreateConfigForm config_type={'testconfig'} existing_names={['existing_config']} submit={onSaveClick} />))
        
        const name_textfield = wrapper.find('[id="name"]')
        name_textfield.instance().value = 'new_config'
        name_textfield.simulate('change', '[id="name"]')

        const description_textfield = wrapper.find('[id="description"]')
        description_textfield.instance().value = 'new_description'
        description_textfield.simulate('change', '[id="description"]')

        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)

        expect(onSaveClick.args[0][0]).to.deep.equal({name: 'new_config', description: 'new_description'})
    })

    it('warning on create with empty configuration name', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<CreateConfigForm config_type={'testconfig'} existing_names={[]} submit={onSaveClick} />))

        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 0)

        expect(wrapper.html()).to.contain('Cannot create a testconfig without a name')
    })

    it('warning on create with existing configuration name', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<CreateConfigForm config_type={'testconfig'} existing_names={['existing_config']} submit={onSaveClick} />))
        
        const name_textfield = wrapper.find('[id="name"]')
        name_textfield.instance().value = 'existing_config'
        name_textfield.simulate('change', '[id="name"]')

        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 0)
        expect(wrapper.html()).to.contain('There is already a configuration with the name existing_config')        
    })

    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<CreateConfigForm config_type={'testconfig'} existing_names={[]} cancel={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<CreateConfigForm config_type={'testconfig'} existing_names={[]} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
