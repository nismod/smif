import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount} from 'enzyme'
import {describe, it} from 'mocha'

import JobRunControls from 'components/Simulation/JobRunControls'

describe('<JobRunControls />', () => {

    it('renders JobRunControls', () => {
        let wrapper = mount(<JobRunControls name='my_name' output='my_output' status='stopped'/>)

        expect(wrapper.state().verbosity).to.equal(0)
        expect(wrapper.state().warm_start).to.equal(false)
        expect(wrapper.state().output_format).to.equal('local_binary')

        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.include('active')

        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(1).html()).to.include('active')

        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).html()).to.include('active')

        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).html()).to.include('Binary')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).html()).to.include('CSV')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).html()).to.not.include('active')
    })

    it('change verbosity', () => {
        let wrapper = mount(<JobRunControls name='my_name' output='my_output' status='stopped'/>)

        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_info"]').find('button').at(0).simulate('click')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.state().verbosity).to.equal(1)

        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).simulate('click')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.state().verbosity).to.equal(2)

        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('active')
        wrapper.find('[id="btn_toggle_info"]').find('button').at(1).simulate('click')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.state().verbosity).to.equal(0)

        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).simulate('click')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.state().verbosity).to.equal(2)

        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(1).html()).to.not.include('active')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.include('active')
        wrapper.find('[id="btn_toggle_debug"]').find('button').at(1).simulate('click')
        expect(wrapper.find('[id="btn_toggle_info"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.find('[id="btn_toggle_debug"]').find('button').at(0).html()).to.not.include('active')
        expect(wrapper.state().verbosity).to.equal(1)
    })

    it('change warm_start', () => {
        let wrapper = mount(<JobRunControls name='my_name' output='my_output' status='stopped'/>)

        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).html()).to.include('ON')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).simulate('click')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.state().warm_start).to.equal(true)

        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).html()).to.include('OFF')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).simulate('click')
        expect(wrapper.find('[id="btn_toggle_warm_start"]').find('button').at(1).html()).to.include('active')
        expect(wrapper.state().warm_start).to.equal(false)
    })

    it('change warm_start', () => {
        let wrapper = mount(<JobRunControls name='my_name' output='my_output' status='stopped'/>)
        
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).html()).to.include('CSV')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).simulate('click')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(1).html()).to.include('active')
        expect(wrapper.state().output_format).to.equal('local_csv')

        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).html()).to.include('Binary')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).html()).to.not.include('active')
        wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).simulate('click')
        expect(wrapper.find('[id="btn_toggle_output_format"]').find('button').at(0).html()).to.include('active')
        expect(wrapper.state().output_format).to.equal('local_binary')
    })
})