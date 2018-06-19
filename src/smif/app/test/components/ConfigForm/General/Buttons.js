import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import {
    CreateButton,
    SaveButton,
    CancelButton,
    DangerButton
} from '../../../../src/components/ConfigForm/General/Buttons'


describe('<CreateButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<CreateButton />).html()
        expect(html).to.not.contain('id="')
        expect(html).to.contain('Add')
    })
    it('renders options', () => {
        const html = shallow(<CreateButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})

describe('<SaveButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<SaveButton />).html()
        expect(html).to.not.contain('id="')
        expect(html).to.contain('Save')
    })
    it('renders options', () => {
        const html = shallow(<SaveButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})

describe('<CancelButton />', () => {
    it('has sensible defaults', () => {
        const html = shallow(<CancelButton />).html()
        expect(html).to.not.contain('id="')
        expect(html).to.contain('Cancel')
    })
    it('renders options', () => {
        const html = shallow(<CancelButton
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
        expect(html).to.contain('Delete')
    })
    it('renders options', () => {
        const html = shallow(<DangerButton
            id="test"
            value="Test" />).html()
        expect(html).to.contain('id="test"')
        expect(html).to.contain('Test')
    })
})
