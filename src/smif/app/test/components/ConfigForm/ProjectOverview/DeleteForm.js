import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'

import DeleteForm from '../../../../src/components/ConfigForm/General/DeleteForm.js'

describe('<DeleteForm />', () => {

    it('renders config_type', () => {
        const form = shallow(<DeleteForm config_name={'testname'} config_type={'testconfig'} existing_names={[]} />)
        expect(form.html()).to.contain('Would you like to delete the <b>testconfig</b> with name <b>testname</b>?')
    })

    it('submit callback on deleteButton click', () => {
        const onDeleteClick = sinon.spy()
        const wrapper = mount((<DeleteForm config_name={'testname'} config_type={'testconfig'} submit={onDeleteClick} />))
    
        wrapper.find('[id="deleteButton"]').simulate('click')
        expect(onDeleteClick).to.have.property('callCount', 1)
    })

    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<DeleteForm config_name={'testname'} config_type={'testconfig'} cancel={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<DeleteForm config_name={'testname'} config_type={'testconfig'} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
