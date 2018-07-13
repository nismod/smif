import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount} from 'enzyme'
import {describe, it} from 'mocha'

import Stepper from 'components/Simulation/Stepper'

describe('<Stepper />', () => {

    it('step unstarted', () => {
        let wrapper = mount(<Stepper status='unstarted'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-process')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-wait')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-wait')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-wait')
    })

    it('step queing', () => {
        let wrapper = mount(<Stepper status='queing'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-process')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-wait')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-wait')
    })

    it('step running', () => {
        let wrapper = mount(<Stepper status='running'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-process')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-wait')
    })

    it('step stopped', () => {
        let wrapper = mount(<Stepper status='stopped'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-error')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-wait')
    })

    it('step done', () => {
        let wrapper = mount(<Stepper status='done'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-process')
    })

    it('step failed', () => {
        let wrapper = mount(<Stepper status='failed'/>)
        expect(wrapper.find('[id="step_ready"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_queuing"]').at(0).html()).to.include('rc-steps-item-finish')
        expect(wrapper.find('[id="step_running"]').at(0).html()).to.include('rc-steps-item-error')
        expect(wrapper.find('[id="step_completed"]').at(0).html()).to.include('rc-steps-item-wait')
    })
})