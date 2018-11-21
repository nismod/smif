import React from 'react'
import {expect} from 'chai'
import {shallow} from 'enzyme'
import {describe, it} from 'mocha'
import {
    SuccessButton,
    PrimaryButton,
    SecondaryButton,
    DangerButton
} from '../../../../src/components/ConfigForm/General/Buttons'


describe('<SuccessButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<SuccessButton />).html()
        expect(html).to.not.contain('id="')
    })
    it('renders options', () => {
        const html = shallow(<SuccessButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})

describe('<PrimaryButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<PrimaryButton />).html()
        expect(html).to.not.contain('id="')
    })
    it('renders options', () => {
        const html = shallow(<PrimaryButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})

describe('<SecondaryButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<SecondaryButton />).html()
        expect(html).to.not.contain('id="')
    })
    it('renders options', () => {
        const html = shallow(<SecondaryButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})

describe('<DangerButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<DangerButton />).html()
        expect(html).to.not.contain('id="')
    })
    it('renders options', () => {
        const html = shallow(<DangerButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})
