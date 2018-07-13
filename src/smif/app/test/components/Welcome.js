import React from 'react'
import { expect } from 'chai'
import { render } from 'enzyme'
import { describe, it } from 'mocha'

import Welcome from '../../src/components/Welcome.js'

describe('<Welcome />', () => {
    it('renders heading text', () => {
        const wrapper = render(<Welcome />)
        expect(wrapper.find('h1').text()).to.contain('Welcome')
    })
})
