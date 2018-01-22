import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import Welcome from '../../src/components/Welcome.js'

describe('<Welcome />', () => {
    it('renders heading text', () => {
        const wrapper = shallow(<Welcome />)
        expect(wrapper.find('h1').text()).to.contain('Welcome')
    })
})
