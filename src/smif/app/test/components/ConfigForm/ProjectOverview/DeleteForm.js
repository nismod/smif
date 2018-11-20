import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount, shallow} from 'enzyme'
import {describe, it} from 'mocha'
import {MemoryRouter} from 'react-router-dom'

import DeleteForm from '../../../../src/components/ConfigForm/General/DeleteForm.js'

describe.skip('<DeleteForm />', () => {

    it('renders config_type', () => {
        const form = shallow(<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'}
            existing_names={[]} />)
        expect(form.html()).to.contain('Would you like to delete the <b>testconfig</b> with name <b>testname</b>?')
    })

    it('refuses if the configuration is in use', () => {
        const form = shallow(<MemoryRouter>
            <DeleteForm
                config_name={'testname'}
                config_type={'testconfig'}
                existing_names={[]}
                in_use_by={[
                    {
                        name: 'Test',
                        link: 'test_link',
                        type: 'test_type'
                    }
                ]} />
        </MemoryRouter>)
        expect(form.html()).to.contain('It is not possible to delete <b>testconfig</b>')
    })

    it('links to the configuration dependency', () => {
        const form = mount(<MemoryRouter>
            <DeleteForm
                config_name={'testname'}
                config_type={'testconfig'}
                existing_names={[]}
                in_use_by={[
                    {
                        name: 'Test',
                        link: 'test_link/',
                        type: 'test_type'
                    }
                ]} />
        </MemoryRouter>)
        const link = form.find('a[href*="test_link/Test"]')
        expect(link.length).to.equal(1)
    })

    it('submit callback on deleteButton click', () => {
        const onDeleteClick = sinon.spy()
        const wrapper = mount((<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'}
            submit={onDeleteClick} />))

        wrapper.find('input#deleteButton').simulate('click')
        expect(onDeleteClick).to.have.property('callCount', 1)
    })

    it('submit on ENTER keypress', () => {
        const onDelete = sinon.spy()
        const wrapper = mount(<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'}
            submit={onDelete} />)

        wrapper.instance().handleKeyPress({keyCode: 13})
        expect(onDelete.calledOnce).to.be.true
    })

    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'}
            cancel={onCancelClick} />))

        wrapper.find('input#cancelDelete').simulate('click')
        expect(onCancelClick.calledOnce).to.be.true
    })

    it('cancel on ESC keypress', () => {
        const onCancel = sinon.spy()
        const wrapper = mount(<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'}
            cancel={onCancel} />)

        wrapper.instance().handleKeyPress({keyCode: 27})
        expect(onCancel.calledOnce).to.be.true
    })

    it('unmount', () => {
        var wrapper = mount(<DeleteForm
            config_name={'testname'}
            config_type={'testconfig'} />)

        wrapper = wrapper.unmount()
        expect(wrapper.length).to.equal(0)
    })
})
