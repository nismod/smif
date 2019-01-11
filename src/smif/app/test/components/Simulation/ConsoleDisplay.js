import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount} from 'enzyme'
import {describe, it} from 'mocha'

import ConsoleDisplay from '../../../src/components/Simulation/ConsoleDisplay'

describe('<ConsoleDisplay />', () => {

    it('renders ConsoleDisplay', () => {
        let wrapper = mount(<ConsoleDisplay name='my_name' output='my_output' status='stopped'/>)

        expect(wrapper.html()).to.contain('my_output')
    })

    it('Scroll down on click', () => {
        let wrapper = mount(<ConsoleDisplay name='my_name' output='my_output' status='running'/>)

        expect(wrapper.state().followConsole).equal(false)
        wrapper.find('[id="btn_toggle_scroll"]').simulate('click')
        expect(wrapper.state().followConsole).equal(true)
        wrapper.find('[id="btn_toggle_scroll"]').simulate('click')
        expect(wrapper.state().followConsole).equal(false)
    })

    it('Download file', () => {
        sinon.spy(ConsoleDisplay.prototype, 'download')

        let wrapper = mount(<ConsoleDisplay name='my_name' output='my_output' status='stopped'/>)
        wrapper.find('[id="btn_download"]').simulate('click')
        expect(ConsoleDisplay.prototype.download).to.have.property('callCount', 1)
        expect(ConsoleDisplay.prototype.download.getCall(0).args[0]).to.contain('my_name')
        expect(ConsoleDisplay.prototype.download.getCall(0).args[1]).to.contain('my_output')
        ConsoleDisplay.prototype.download.restore()
    })
})
